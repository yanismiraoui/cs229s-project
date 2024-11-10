import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import logging
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
import json
import pathlib
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress fvcore warnings
logging.getLogger('fvcore.nn.jit_analysis').setLevel(logging.ERROR)

def get_system_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB
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

def evaluate_model(model, tokenizer, test_data, device, config):
    model.eval()
    total_loss = 0
    total_similarity_loss = 0
    total_flops = 0
    start_time = time.time()
    batch_size = config['evaluation']['batch_size']
    max_length = config['model']['max_length']
    
    # Calculate total number of batches
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    logger.info(f"Starting evaluation with {num_batches} batches...")
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_start_time = time.time()
            current_batch = i // batch_size + 1
            
            batch_texts = test_data[i:i + batch_size]
            
            # Process the entire batch at once
            prompts = [f"<|assistant|>{text}<|user|>" for text in batch_texts]
            batch_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
                pad_to_multiple_of=8  # Optional: for better performance
            )
            
            input_ids = batch_inputs['input_ids'].to(device)
            attention_mask = batch_inputs['attention_mask'].to(device)
            
            # Count FLOPS for this batch
            flops = FlopCountAnalysis(model, (input_ids, attention_mask))
            batch_flops = flops.total()
            total_flops += batch_flops
            
            # Get model outputs with hidden states
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True  # Get hidden states
            )
            
            loss = outputs.loss
            
            # Compute similarity loss between consecutive layers
            hidden_states = outputs.hidden_states
            similarity_losses = []
            for j in range(len(hidden_states)-1):
                sim_loss = compute_similarity_loss(
                    hidden_states[j],
                    hidden_states[j+1],
                    attention_mask
                )
                similarity_losses.append(sim_loss)
            
            # Average similarity loss across layers
            avg_similarity_loss = torch.stack(similarity_losses).mean()
            
            total_loss += loss.item()
            total_similarity_loss += avg_similarity_loss.item()
            
            # Calculate time estimates
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / current_batch
            remaining_batches = num_batches - current_batch
            estimated_time_remaining = remaining_batches * avg_time_per_batch
            
            # Measure resources and log
            cpu_percent, memory_used = get_system_resources()
            logger.info(
                f"Batch {current_batch}/{num_batches} "
                f"({(current_batch/num_batches)*100:.1f}%): "
                f"CPU: {cpu_percent}%, Mem: {memory_used:.2f}MB, "
                f"Loss: {loss.item():.4f}, Sim Loss: {avg_similarity_loss.item():.4f}, "
                f"FLOPS: {batch_flops/1e9:.2f}G\n"
                f"Time: {batch_time:.2f}s/batch, "
                f"ETA: {estimated_time_remaining/60:.1f}min remaining"
            )
    
    end_time = time.time()
    wall_time = end_time - start_time
    avg_loss = total_loss * batch_size / len(test_data)
    avg_similarity_loss = total_similarity_loss * batch_size / len(test_data)
    avg_flops = total_flops / len(test_data)
    
    return {
        'avg_loss': avg_loss,
        'avg_similarity_loss': avg_similarity_loss,
        'wall_time': wall_time,
        'total_flops': total_flops,
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
    
    # Combine and clean the data
    test_data = [line.strip() for line in natural_language + commands if line.strip()]
    
    # Calculate how many examples to use based on eval_size
    eval_size = config['data'].get('eval_size', 1.0)  # Default to using all data if not specified
    num_examples = int(len(test_data) * eval_size)
    
    # Take the first num_examples
    test_data = test_data[:num_examples]
    
    logger.info(f"Using {eval_size*100}% of data ({num_examples} examples) for evaluation")
    
    return test_data

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CPU usage
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load the model and tokenizer
    model_name = config['model']['base_model']
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    model = model.to(device)
    
    # Load test data
    test_data = load_test_data(config)
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Start evaluation
    logger.info("Starting evaluation...")
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    results = evaluate_model(model, tokenizer, test_data, device, config)
    
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memory_used = end_memory - start_memory
    
    # Create results directory if it doesn't exist
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare results dictionary
    results_dict = {
        "model": {
            "name": model_name,
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
            "num_test_examples": len(test_data),
            "total_examples_available": len(test_data) / config['data'].get('eval_size', 1.0)
        },
        "metrics": {
            "average_loss": float(results['avg_loss']),
            "average_similarity_loss": float(results['avg_similarity_loss']),
            "wall_time_seconds": float(results['wall_time']),
            "total_memory_mb": float(memory_used),
            "total_flops_g": float(results['total_flops']/1e9),
            "average_flops_per_sample_g": float(results['avg_flops']/1e9)
        },
        "runtime": {
            "device": "CPU",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Log summary to console
    logger.info(f"\nEvaluation Results Summary:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Data: Using {config['data'].get('eval_size', 1.0)*100:.1f}% of available data ({len(test_data)} of {int(len(test_data)/config['data'].get('eval_size', 1.0))} examples)")
    logger.info(f"Quantization: {'Enabled (' + str(config['model']['quantization']['bits']) + ' bits)' if config['model']['quantization']['enabled'] else 'Disabled'}")
    logger.info(f"Pruning: {'Enabled (sparsity ' + str(config['model']['pruning']['target_sparsity']) + ')' if config['model']['pruning']['enabled'] else 'Disabled'}")
    logger.info(f"Average Loss: {results['avg_loss']:.4f}")
    logger.info(f"Average Similarity Loss: {results['avg_similarity_loss']:.4f}")
    logger.info(f"Wall Time: {results['wall_time']:.2f} seconds")
    logger.info(f"Total Memory Used: {memory_used:.2f} MB")
    logger.info(f"Total FLOPS: {results['total_flops']/1e9:.2f}G")
    logger.info(f"Average FLOPS per sample: {results['avg_flops']/1e9:.2f}G")

if __name__ == "__main__":
    main()