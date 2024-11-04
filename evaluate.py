import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def evaluate_model(model, tokenizer, test_data, device, config):
    model.eval()
    total_loss = 0
    start_time = time.time()
    batch_size = config['evaluation']['batch_size']
    max_length = config['model']['max_length']
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_texts = test_data[i:i + batch_size]
            
            # Process the entire batch at once instead of individual examples
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
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=input_ids  # Using input_ids as labels for calculating loss
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Measure resources during evaluation
            cpu_percent, memory_used = get_system_resources()
            logger.info(f"Batch {i//batch_size + 1}: CPU Usage: {cpu_percent}%, Memory Used: {memory_used:.2f} MB, Loss: {loss.item()}, Wall Time: {time.time() - start_time:.2f} seconds")
    
    end_time = time.time()
    wall_time = end_time - start_time
    avg_loss = total_loss * batch_size / len(test_data)
    
    return {
        'avg_loss': avg_loss,
        'wall_time': wall_time
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
    
    # Log results
    logger.info(f"\nEvaluation Results:")
    logger.info(f"Average Loss: {results['avg_loss']:.4f}")
    logger.info(f"Wall Time: {results['wall_time']:.2f} seconds")
    logger.info(f"Total Memory Used: {memory_used:.2f} MB")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"================\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Average Loss: {results['avg_loss']:.4f}\n")
        f.write(f"Wall Time: {results['wall_time']:.2f} seconds\n")
        f.write(f"Total Memory Used: {memory_used:.2f} MB\n")
        f.write(f"Device: CPU\n")
        f.write(f"Number of test examples: {len(test_data)}\n")

if __name__ == "__main__":
    main()