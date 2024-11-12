import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import yaml
import logging
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
import json
import pathlib
from torch.nn import functional as F
from models.model import BoltModel
from Levenshtein import distance as levenshtein_distance

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress fvcore warnings
logging.getLogger('fvcore.nn.jit_analysis').setLevel(logging.ERROR)

def get_system_resources():
    # Get CPU usage over a 2-second interval for more accurate reading
    cpu_percent = psutil.cpu_percent(interval=2)
    # Get memory usage for the current process
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024  # in MB
    return cpu_percent, memory

def prepare_input(text, tokenizer, max_length):
    # Prepare the input following DeepSeek's format
    prompt = f"<|assistant|>{text}<|user|>"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding='max_length',  # Changed to force all inputs to same length
    )
    return inputs

def compute_similarity_loss(hidden_states1, hidden_states2, attention_mask=None):
    """Compute cosine similarity between hidden states"""
    # Get the last hidden states
    last_hidden1 = hidden_states1[:, -1, :]
    last_hidden2 = hidden_states2[:, -1, :]
    
    # Normalize the vectors
    last_hidden1 = F.normalize(last_hidden1, p=2, dim=1)
    last_hidden2 = F.normalize(last_hidden2, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(last_hidden1 * last_hidden2, dim=1)
    
    # Convert to loss (1 - similarity)
    similarity_loss = 1 - similarity.mean()
    
    return similarity_loss

def compute_levenshtein_distance(generated_text, target_text):
    """Compute the Levenshtein distance between generated and target text"""
    logger.info(f"Generated: {generated_text}")
    logger.info(f"Target: {target_text}")
    return levenshtein_distance(generated_text, target_text)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids_list):
        self.stop_ids_list = stop_ids_list

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        for stop_ids in self.stop_ids_list:
            if len(stop_ids) > len(input_ids[0]):
                continue
            if torch.equal(input_ids[0][-len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                return True
        return False

def evaluate_model(model, tokenizer, test_data, device, config):
    model.eval()
    total_loss = 0
    total_similarity_loss = 0
    total_flops = 0
    total_levenshtein = 0
    start_time = time.time()
    # Get initial resource usage
    initial_cpu, initial_memory = get_system_resources()
    
    batch_size = config['evaluation']['batch_size']
    max_length = config['model']['max_length']
    
    num_batches = (len(test_data['natural_language']) + batch_size - 1) // batch_size
    logger.info(f"Starting evaluation with {num_batches} batches...")
    
    with torch.no_grad():
        for i in range(0, len(test_data['natural_language']), batch_size):
            batch_start_time = time.time()
            current_batch = i // batch_size + 1
            
            batch_texts = test_data['natural_language'][i:i + batch_size]
            batch_commands = test_data['commands'][i:i + batch_size]
            
            # Process the entire batch at once
            prompts = [f"You are a helpful bash command assistant. The user asked: {text}\n [BEGIN COMMAND]\n" for text in batch_texts]
            batch_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
                pad_to_multiple_of=8
            )
            
            input_ids = batch_inputs['input_ids'].to(device)
            attention_mask = batch_inputs['attention_mask'].to(device)
            
            # Try to count FLOPS
            try:
                flops = FlopCountAnalysis(model, (input_ids, attention_mask))
                batch_flops = flops.total()
                total_flops += batch_flops
                flops_message = f"FLOPS: {batch_flops/1e9:.2f}G"
            except Exception as e:
                logger.warning(f"Failed to count FLOPS: {str(e)}")
                batch_flops = 0
                flops_message = "FLOPS: N/A"

            # Set stopping criteria
            stopping_list = ["[END COMMAND]", "[END]"]
            stop_ids_list = [tokenizer.encode(word, add_special_tokens=False) for word in stopping_list]
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids_list)])
            
            # Generate outputs first
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,  # Get hidden states during generation
                # stopping_criteria=stopping_criteria
            )
            
            generated_sequences = generated_outputs.sequences
            attention_mask_generated = torch.ones_like(generated_sequences)

            # Get loss and hidden states from the generated sequence
            outputs = model(
                input_ids=generated_sequences,
                attention_mask=attention_mask_generated,
                labels=generated_sequences,
                output_hidden_states=True,
            )
            
            loss = outputs.loss
            hidden_states = outputs.hidden_states
            
            # Compute similarity loss
            similarity_losses = []
            for j in range(len(hidden_states)-1):
                sim_loss = compute_similarity_loss(
                    hidden_states[j],
                    hidden_states[j+1],
                    attention_mask
                )
                similarity_losses.append(sim_loss)
            
            avg_similarity_loss = torch.stack(similarity_losses).mean()
            
            total_loss += loss.item()
            total_similarity_loss += avg_similarity_loss.item()
            
            # Decode generated outputs and compute Levenshtein distance
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            # remove the prompt from the generated text
            # generated_texts = [text.replace(prompt, "") for text, prompt in zip(generated_texts, prompts)]
            # remove the stop tokens from the generated text
            # generated_texts = [text.replace("[END COMMAND]", "").replace("[END]", "") for text in generated_texts]
            
            batch_levenshtein = 0
            for gen_text, target_text in zip(generated_texts, batch_commands):
                gen_text = gen_text.replace("<|assistant|>", "").replace("<|user|>", "").strip()
                batch_levenshtein += compute_levenshtein_distance(gen_text, target_text)
            total_levenshtein += batch_levenshtein
            
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / current_batch
            remaining_batches = num_batches - current_batch
            estimated_time_remaining = remaining_batches * avg_time_per_batch
            
            # Get resource usage for this batch
            current_cpu, current_memory = get_system_resources()
            memory_used = current_memory - initial_memory
            logger.info(
                f"Batch {current_batch}/{num_batches} "
                f"({(current_batch/num_batches)*100:.1f}%): "
                f"CPU: {current_cpu}%, Mem Change: {memory_used:.2f}MB, "
                f"Loss: {loss.item():.4f}, Sim Loss: {avg_similarity_loss.item():.4f}, "
                f"Avg Levenshtein: {batch_levenshtein/len(batch_commands):.2f}, "
                f"{flops_message}\n"
                f"Time: {batch_time:.2f}s/batch, "
                f"ETA: {estimated_time_remaining/60:.1f}min remaining"
            )
    
    # Calculate final metrics
    end_time = time.time()
    wall_time = end_time - start_time
    avg_loss = total_loss * batch_size / len(test_data)
    avg_similarity_loss = total_similarity_loss * batch_size / len(test_data)
    avg_flops = total_flops / len(test_data) if total_flops > 0 else None
    avg_levenshtein = total_levenshtein / len(test_data)
    
    return {
        'avg_loss': avg_loss,
        'avg_similarity_loss': avg_similarity_loss,
        'avg_levenshtein': avg_levenshtein,
        'wall_time': wall_time,
        'total_flops': total_flops if total_flops > 0 else None,
        'avg_flops': avg_flops
    }

def load_test_data(config):
    # Load your test data from files
    test_data = []
    
    # Load natural language examples
    with open(config['data']['natural_language_file'], 'r') as f:
        natural_language = f.readlines()
    
    # Load command examples
    with open(config['data']['command_file'], 'r') as f:
        commands = f.readlines()
    
    # Combine the data into a single dict
    test_data = {
        'natural_language': natural_language,
        'commands': commands
    }
    
    # Calculate how many examples to use based on eval_size
    eval_size = config['data'].get('eval_size', 1.0)  # Default to using all data if not specified
    num_examples = int(len(test_data['natural_language']) * eval_size)
    
    # Take the first num_examples
    test_data = {key: test_data[key][:num_examples] for key in test_data}
    
    logger.info(f"Using {eval_size*100}% of data ({num_examples} examples) for evaluation")
    
    return test_data

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CPU usage
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Use BoltModel instead of direct model loading
    bolt_model = BoltModel(config)
    model, tokenizer = bolt_model.get_model_and_tokenizer()
    
    model = model.to(device)

    # Load assistant model if enabled
    if config['model']['assisted_decoding']['enabled']:
        logger.info(f"Using assisted decoding with model: {config['model']['assisted_decoding']['model']}")
        logger.info(f"Loading assistant model...")
        assistant_model = AutoModelForCausalLM.from_pretrained(
            config['model']['assisted_decoding']['model'],
            low_cpu_mem_usage=True,
            trust_remote_code=True
            )
        assistant_model = assistant_model.to(device)
    else:
        assistant_model = None
    
    # Load test data
    test_data = load_test_data(config)
    logger.info(f"Loaded {len(test_data['natural_language'])} test examples")
    
    # Start evaluation
    logger.info("Starting evaluation...")
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    results = evaluate_model(model, tokenizer, test_data, device, config)
    
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memory_used = end_memory - start_memory
    
    # Create results directory if it doesn't exist
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare results dictionary with safe FLOPS handling
    results_dict = {
        "model": {
            "name": config['model']['base_model'],
            "configuration": {
                "quantization": {
                    "enabled": config['model']['quantization']['enabled'],
                    "bits": config['model']['quantization']['bits'] if config['model']['quantization']['enabled'] else None
                },
                "pruning": {
                    "enabled": config['model']['pruning']['enabled'],
                    "target_sparsity": config['model']['pruning']['target_sparsity'] if config['model']['pruning']['enabled'] else None
                }
            }
        },
        "data": {
            "eval_fraction": config['data'].get('eval_size', 1.0),
            "num_test_examples": len(test_data['natural_language']),
            "total_examples_available": len(test_data['natural_language']) / config['data'].get('eval_size', 1.0)
        },
        "metrics": {
            "average_loss": float(results['avg_loss']),
            "average_similarity_loss": float(results['avg_similarity_loss']),
            "wall_time_seconds": float(results['wall_time']),
            "total_memory_mb": float(memory_used)
        },
        "runtime": {
            "device": "CPU",
            "timestamp": datetime.now().isoformat()
        }
    }

    # Add FLOPS metrics only if they were successfully calculated
    if results['total_flops'] is not None:
        results_dict["metrics"].update({
            "total_flops_g": float(results['total_flops']/1e9),
            "average_flops_per_sample_g": float(results['avg_flops']/1e9)
        })
    else:
        results_dict["metrics"].update({
            "total_flops_g": None,
            "average_flops_per_sample_g": None
        })
    
    # Add Levenshtein distance metrics
    results_dict["metrics"].update({
        "average_levenshtein_distance": float(results['avg_levenshtein'])
    })
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Log summary to console
    logger.info(f"\nEvaluation Results Summary:")
    logger.info(f"Model: {config['model']['base_model']}")
    logger.info(f"Data: Using {config['data'].get('eval_size', 1.0)*100:.1f}% of available data ({len(test_data)} of {int(len(test_data)/config['data'].get('eval_size', 1.0))} examples)")
    logger.info(f"Quantization: {'Enabled (' + str(config['model']['quantization']['bits']) + ' bits)' if config['model']['quantization']['enabled'] else 'Disabled'}")
    logger.info(f"Pruning: {'Enabled (sparsity ' + str(config['model']['pruning']['target_sparsity']) + ')' if config['model']['pruning']['enabled'] else 'Disabled'}")
    logger.info(f"Average Loss: {results['avg_loss']:.4f}")
    logger.info(f"Average Similarity Loss: {results['avg_similarity_loss']:.4f}")
    logger.info(f"Wall Time: {results['wall_time']:.2f} seconds")
    logger.info(f"Total Memory Used: {memory_used:.2f} MB")
    if results['total_flops'] is not None:
        logger.info(f"Total FLOPS: {results['total_flops']/1e9:.2f}G")
        logger.info(f"Average FLOPS per sample: {results['avg_flops']/1e9:.2f}G")
    else:
        logger.info("FLOPS calculation failed")
    logger.info(f"Average Levenshtein Distance: {results['avg_levenshtein']:.2f}")

if __name__ == "__main__":
    main()