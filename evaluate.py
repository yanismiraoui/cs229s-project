import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, StoppingCriteriaList
import yaml
import logging
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
import json
import pathlib
from models.model import BoltModel
from evaluators.openai_evaluator import setup_openai, batch_validate_with_gpt4
from utils.evaluation_utils import (
    get_system_resources,
    compute_similarity_loss,
    compute_levenshtein_distance,
    StopOnTokens,
    load_test_data
)
from utils.command_validator import batch_validate_commands

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress fvcore warnings
logging.getLogger('fvcore.nn.jit_analysis').setLevel(logging.ERROR)

def evaluate_model(model, tokenizer, test_data, device, config, memory_tracker, start_memory):
    
    model.eval()
    total_loss = 0
    total_similarity_loss = 0
    total_flops = 0
    total_levenshtein = 0
    total_exact_matches = 0
    start_time = time.time()
    # Get initial resource usage
    initial_cpu, initial_memory = get_system_resources()
    
    batch_size = config['evaluation']['batch_size']
    max_length = config['model']['max_length']
    
    num_batches = (len(test_data['natural_language']) + batch_size - 1) // batch_size
    logger.info(f"Starting evaluation with {num_batches} batches...")
    
    # Initialize GPT validation metrics if enabled
    gpt_validation_enabled = setup_openai()
    total_gpt_score = 0
    total_gpt_validations = 0
    
    # Initialize syntax validation metrics
    total_syntax_valid = 0
    total_syntax_checks = 0
    syntax_errors = []
    
    with torch.no_grad():
        for i in range(0, len(test_data['natural_language']), batch_size):
            batch_start_time = time.time()
            current_batch = i // batch_size + 1
            
            batch_texts = test_data['natural_language'][i:i + batch_size]
            batch_commands = test_data['commands'][i:i + batch_size]
            
            # Process the entire batch at once
            prompts = [f"You are a helpful bash command assistant. The user asked: {text}\nProvide only the command and end with [END COMMAND].\n[BEGIN COMMAND] " for text in batch_texts]
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
                # Count FLOPS for forward pass only
                forward_flops = FlopCountAnalysis(model, (input_ids, attention_mask))
                forward_flops_count = forward_flops.total()
                
                max_new_tokens = config['model'].get('max_new_tokens', 50)
                generation_flops = sum([forward_flops_count * (i/input_ids.shape[1]) 
                                     for i in range(max_new_tokens)])
                
                batch_flops = forward_flops_count + generation_flops
                total_flops += batch_flops
                flops_message = (
                    f"FLOPS: {batch_flops/1e9:.2f}G "
                    f"(Forward: {forward_flops_count/1e9:.2f}G, "
                    f"Generation: {generation_flops/1e9:.2f}G)"
                )
            except Exception as e:
                logger.warning(f"Failed to count FLOPS: {str(e)}")
                batch_flops = 0
                flops_message = "FLOPS: N/A"

            # Enhanced stopping criteria
            stopping_list = ["END COMMAND", "END"]
            stop_ids_list = []
            for word in stopping_list:
                # Handle both single token and multi-token cases
                tokens = tokenizer.encode(word, add_special_tokens=False)
                if len(tokens) > 0:
                    stop_ids_list.append(tokens)
                # Also add the token with a space prefix
                # space_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
                # if len(space_tokens) > 0:
                #     stop_ids_list.append(space_tokens)
            
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids_list)])
            
            # Generate outputs first
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,  # Get hidden states during generation
                # stopping_criteria=stopping_criteria
            )
            
            generated_sequences = generated_outputs.sequences
            attention_mask_generated = (generated_sequences != tokenizer.pad_token_id).long()

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
                    attention_mask_generated
                )
                similarity_losses.append(sim_loss)
            
            avg_similarity_loss = torch.stack(similarity_losses).mean()
            
            total_loss += loss.item()
            total_similarity_loss += avg_similarity_loss.item()
            
            # Decode generated outputs and compute Levenshtein distance
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            # Remove the prompt and stop tokens
            print(generated_texts)
            generated_texts = [
                text.split("[BEGIN COMMAND]")[-1]
                    .split("[END COMMAND]")[0]  # Try full end token first
                    .split("[END]")[0]  # Try shorter end token
                    .split("[END ")[0]  # Handle partial end token
                    .strip()
                for text in generated_texts
            ]
            
            batch_levenshtein = 0
            for gen_text, target_text in zip(generated_texts, batch_commands):
                gen_text = gen_text.replace("<|assistant|>", "").replace("<|user|>", "").strip()
                batch_levenshtein += compute_levenshtein_distance(gen_text, target_text)
            total_levenshtein += batch_levenshtein
            
            # Count exact matches in batch
            batch_exact_matches = 0
            for gen_text, target_text in zip(generated_texts, batch_commands):
                gen_text = gen_text.replace("<|assistant|>", "").replace("<|user|>", "").strip()
                if gen_text == target_text.strip():
                    batch_exact_matches += 1
            total_exact_matches += batch_exact_matches
            
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
                f"Exact Matches: {batch_exact_matches}/{len(batch_commands)}, "
                f"Avg Levenshtein: {batch_levenshtein/len(batch_commands):.2f}, "
                f"{flops_message}\n"
                f"Time: {batch_time:.2f}s/batch, "
                f"ETA: {estimated_time_remaining/60:.1f}min remaining"
            )
            
            # Validate all commands with GPT-4
            if gpt_validation_enabled:
                try:
                    # Use min of batch_size and 64 for GPT validation
                    gpt_batch_size = min(len(generated_texts), 64)
                    validation_results = batch_validate_with_gpt4(
                        generated_texts,
                        batch_commands,
                        batch_texts,
                        batch_size=gpt_batch_size
                    )
                    
                    for gen_text, target_text, result in zip(generated_texts, batch_commands, validation_results):
                        total_gpt_score += result.get('correctness_score', 0)
                        total_gpt_validations += 1
                        
                        logger.info("\nGPT-4 Validation:")
                        logger.info(f"Generated: {gen_text}")
                        logger.info(f"Target: {target_text}")
                        logger.info(f"Score: {result.get('correctness_score', 0):.2f}")
                        logger.info(f"Equivalent: {result.get('functionally_equivalent', False)}")
                except Exception as e:
                    logger.warning(f"GPT-4 batch validation failed: {str(e)}")
            
            # Validate generated commands
            validation_results = batch_validate_commands(generated_texts)
            batch_syntax_valid = sum(1 for r in validation_results if r['is_valid'])
            total_syntax_valid += batch_syntax_valid
            total_syntax_checks += len(validation_results)
            
            # Collect syntax errors for logging
            batch_errors = [
                {
                    'command': r['command'],
                    'error': r['error']
                }
                for r in validation_results if not r['is_valid']
            ]
            syntax_errors.extend(batch_errors)
            
            # Log syntax validation results
            logger.info(f"Syntax validation - Valid: {batch_syntax_valid}/{len(batch_commands)} "
                       f"({batch_syntax_valid/len(batch_commands)*100:.1f}%)")
            for result in validation_results:
                if not result['is_valid']:
                    logger.debug(f"Invalid command: {result['command']}")
                    logger.debug(f"Error: {result['error']}")
    
    # Calculate final metrics including exact match accuracy
    exact_match_accuracy = total_exact_matches / len(test_data['natural_language'])
    
    # Calculate final metrics
    end_time = time.time()
    wall_time = end_time - start_time
    avg_loss = total_loss * batch_size / len(test_data['natural_language'])
    avg_similarity_loss = total_similarity_loss * batch_size / len(test_data['natural_language'])
    avg_flops = total_flops / len(test_data['natural_language']) if total_flops > 0 else None
    avg_levenshtein = total_levenshtein / len(test_data['natural_language'])
    
    # Add GPT validation results to return dict
    avg_gpt_score = total_gpt_score / total_gpt_validations if total_gpt_validations > 0 else None
    
    # Track memory at end of evaluation
    end_memory = memory_tracker.memory_info().rss / 1024 / 1024  # MB
    # Replace peak_memory calculation with more accurate tracking
    memory_info = memory_tracker.memory_info()
    peak_memory = max(
        memory_info.rss,  # Resident Set Size
        memory_info.vms   # Virtual Memory Size
    ) / 1024 / 1024  # Convert to MB
    
    # Add syntax validation to return dict
    syntax_validity_rate = total_syntax_valid / total_syntax_checks if total_syntax_checks > 0 else 0
    
    return {
        'avg_loss': avg_loss,
        'avg_similarity_loss': avg_similarity_loss,
        'avg_levenshtein': avg_levenshtein,
        'exact_match_accuracy': exact_match_accuracy,
        'total_exact_matches': total_exact_matches,
        'wall_time': wall_time,
        'total_flops': total_flops if total_flops > 0 else None,
        'avg_flops': avg_flops,
        'gpt_validation_score': avg_gpt_score,
        'gpt_validations_performed': total_gpt_validations,
        'memory': {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'used_memory_mb': end_memory - start_memory,
            'peak_memory_mb': peak_memory,
            'virtual_memory_mb': memory_info.vms / 1024 / 1024,
            'resident_memory_mb': memory_info.rss / 1024 / 1024
        },
        'syntax_validation': {
            'total_valid': total_syntax_valid,
            'total_checked': total_syntax_checks,
            'validity_rate': syntax_validity_rate,
            'errors': syntax_errors[:10]  # Store first 10 errors as examples
        }
    }

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CPU usage
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Enhanced memory tracking
    memory_tracker = psutil.Process(os.getpid())
    memory_info = memory_tracker.memory_info()
    start_memory = memory_info.rss / 1024 / 1024  # MB
    
    # Modified model loading logic
    bolt_model = BoltModel(config)
    
    if config['model']['use_finetuned']:
        logger.info(f"Loading finetuned model from {config['model']['finetuned_path']}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config['model']['finetuned_path'],
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True  # Ensure we only look for local files
            )
            # Keep the tokenizer from the finetuned model path for consistency
            tokenizer = bolt_model.tokenizer
            logger.info("Successfully loaded finetuned model")
        except Exception as e:
            logger.error(f"Failed to load finetuned model: {str(e)}")
            logger.info("Falling back to base model...")
            model, tokenizer = bolt_model.get_model_and_tokenizer()
    else:
        logger.info(f"Using base model: {config['model']['base_model']}")
        model, tokenizer = bolt_model.get_model_and_tokenizer()
    
    model = model.to(device)

    # Load assistant model if enabled
    if config['model']['assisted_decoding']['enabled']:
        logger.info(f"Using assisted decoding with model: {config['model']['assisted_decoding']['model']}")
        logger.info("Loading assistant model...")
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
    results = evaluate_model(model, tokenizer, test_data, device, config, memory_tracker, start_memory)
    
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
            "memory_usage_mb": {
                "start": float(results['memory']['start_memory_mb']),
                "end": float(results['memory']['end_memory_mb']),
                "used": float(results['memory']['used_memory_mb']),
                "peak": float(results['memory']['peak_memory_mb']) if results['memory']['peak_memory_mb'] is not None else None
            }
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
    
    # Add exact match metrics to results
    results_dict["metrics"].update({
        "exact_match_accuracy": float(results['exact_match_accuracy']),
        "total_exact_matches": int(results['total_exact_matches'])
    })
    
    # Add GPT validation metrics to results if available
    if results.get('gpt_validation_score') is not None:
        results_dict["metrics"].update({
            "gpt_validation_score": float(results['gpt_validation_score']),
            "gpt_validations_performed": int(results['gpt_validations_performed'])
        })
    
    # Add syntax validation metrics to results
    if 'syntax_validation' in results:
        results_dict["metrics"].update({
            "syntax_validation": {
                "validity_rate": float(results['syntax_validation']['validity_rate']),
                "total_valid": int(results['syntax_validation']['total_valid']),
                "total_checked": int(results['syntax_validation']['total_checked']),
                "error_examples": results['syntax_validation']['errors']  # Include sample errors
            }
        })
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Log summary to console
    logger.info("\nEvaluation Results Summary:")
    logger.info(f"Model: {config['model']['base_model']}")
    logger.info(f"Data: Using {config['data'].get('eval_size', 1.0)*100:.1f}% of available data ({len(test_data['natural_language'])} of {int(len(test_data['natural_language'])/config['data'].get('eval_size', 1.0))} examples)")
    logger.info(f"Quantization: {'Enabled (' + str(config['model']['quantization']['bits']) + ' bits)' if config['model']['quantization']['enabled'] else 'Disabled'}")
    logger.info(f"Pruning: {'Enabled (sparsity ' + str(config['model']['pruning']['target_sparsity']) + ')' if config['model']['pruning']['enabled'] else 'Disabled'}")
    logger.info(f"Average Loss: {results['avg_loss']:.4f}")
    logger.info(f"Average Similarity Loss: {results['avg_similarity_loss']:.4f}")
    logger.info(f"Wall Time: {results['wall_time']:.2f} seconds")
    logger.info(f"Memory Usage:")
    logger.info(f"  Start: {results['memory']['start_memory_mb']:.2f} MB")
    logger.info(f"  End: {results['memory']['end_memory_mb']:.2f} MB")
    logger.info(f"  Used: {results['memory']['used_memory_mb']:.2f} MB")
    if results['memory']['peak_memory_mb'] is not None:
        logger.info(f"  Peak: {results['memory']['peak_memory_mb']:.2f} MB")
    if results['total_flops'] is not None:
        logger.info(f"Total FLOPS: {results['total_flops']/1e9:.2f}G")
        logger.info(f"Average FLOPS per sample: {results['avg_flops']/1e9:.2f}G")
    else:
        logger.info("FLOPS calculation failed")
    logger.info(f"Average Levenshtein Distance: {results['avg_levenshtein']:.2f}")
    logger.info(f"Exact Matches: {results['total_exact_matches']}/{len(test_data['natural_language'])} ({results['exact_match_accuracy']*100:.2f}%)")
    if results.get('gpt_validation_score') is not None:
        logger.info(f"GPT-4 Validation Score: {results['gpt_validation_score']}")
        logger.info(f"GPT-4 Validations Performed: {results['gpt_validations_performed']}")
    if 'syntax_validation' in results:
        logger.info("\nSyntax Validation Results:")
        logger.info(f"Valid Commands: {results['syntax_validation']['total_valid']}/{results['syntax_validation']['total_checked']} "
                   f"({results['syntax_validation']['validity_rate']*100:.1f}%)")
        if results['syntax_validation']['errors']:
            logger.info("\nSample Syntax Errors:")
            for error in results['syntax_validation']['errors']:
                logger.info(f"Command: {error['command']}")
                logger.info(f"Error: {error['error']}")
                logger.info("---")

if __name__ == "__main__":
    main()