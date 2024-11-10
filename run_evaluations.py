import yaml
import copy
import subprocess
from pathlib import Path
import time

# List of small coding models to evaluate
MODELS = [
    "facebook/opt-125m",
    "Salesforce/codegen-350M-mono",
    "bigcode/tiny_starcoder",
    "bigcode/starcoderbase-1b",  # slightly larger but still manageable
    "microsoft/phi-1",
    "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-ai/deepseek-coder-1.3b-base"
]

# Configuration templates for different optimization settings
OPTIMIZATION_CONFIGS = [
    {
        "name": "baseline",
        "quantization": {"enabled": False},
        "pruning": {"enabled": False, "target_sparsity": 0.3},
        "assisted_decoding": {"enabled": False}
    },
    {
        "name": "quantized",
        "quantization": {"enabled": True, "bits": 8},
        "pruning": {"enabled": False, "target_sparsity": 0.3},
        "assisted_decoding": {"enabled": False}
    },
    {
        "name": "pruned",
        "quantization": {"enabled": False},
        "pruning": {"enabled": True, "target_sparsity": 0.3},
        "assisted_decoding": {"enabled": False}
    },
    {
        "name": "quantized_pruned",
        "quantization": {"enabled": True, "bits": 8},
        "pruning": {"enabled": True, "target_sparsity": 0.3},
        "assisted_decoding": {"enabled": False}
    },
    {
        "name": "assisted_decoding",
        "quantization": {"enabled": False},
        "pruning": {"enabled": False, "target_sparsity": 0.3},
        "assisted_decoding": {"enabled": True, "model": "deepseek-ai/deepseek-coder-1.3b-base"}
    }
]

def load_base_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_config_variant(base_config, model_name, opt_config):
    """Create a config variant for a specific model and optimization setting"""
    config = copy.deepcopy(base_config)
    
    # Update model name
    config['model']['base_model'] = model_name
    
    # Update optimization settings
    config['model']['quantization'] = opt_config['quantization']
    config['model']['pruning'] = opt_config['pruning']
    config['model']['assisted_decoding'] = opt_config['assisted_decoding']
    
    # Set evaluation size to 20%
    config['data']['eval_size'] = 1
    
    return config

def save_config(config, filename):
    """Save config to a temporary file"""
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def run_evaluation(config_path):
    """Run the evaluation script with the given config"""
    try:
        subprocess.run(['python', 'evaluate.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        return False
    return True

def main():
    # Load base configuration
    base_config = load_base_config()
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Track successful and failed evaluations
    results = {
        "successful": [],
        "failed": []
    }
    
    # Iterate over all models and optimization configurations
    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}\n")
        
        for opt_config in OPTIMIZATION_CONFIGS:
            print(f"\nRunning {opt_config['name']} configuration...")
            
            # Create config variant
            config = create_config_variant(base_config, model, opt_config)
            
            # Save temporary config
            save_config(config, 'configs/config.yaml')
            
            # Run evaluation
            start_time = time.time()
            success = run_evaluation('configs/config.yaml')
            duration = time.time() - start_time
            
            # Track results
            eval_result = {
                "model": model,
                "optimization": opt_config['name'],
                "duration": duration
            }
            
            if success:
                results["successful"].append(eval_result)
                print(f"✓ Evaluation completed in {duration:.2f} seconds")
            else:
                results["failed"].append(eval_result)
                print(f"✗ Evaluation failed after {duration:.2f} seconds")
            
            # Small delay between runs
            time.sleep(2)
    
    # Print summary
    print("\n\nEvaluation Summary")
    print("==================")
    print(f"Total evaluations: {len(results['successful']) + len(results['failed'])}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed evaluations:")
        for failure in results['failed']:
            print(f"- {failure['model']} ({failure['optimization']})")

if __name__ == "__main__":
    main() 