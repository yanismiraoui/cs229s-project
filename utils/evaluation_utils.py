import torch
import psutil
import os
from torch.nn import functional as F
from Levenshtein import distance as levenshtein_distance
import logging

logger = logging.getLogger(__name__)

def get_system_resources():
    """Get CPU usage and memory usage for the current process"""
    cpu_percent = psutil.cpu_percent(interval=2)
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024  # in MB
    return cpu_percent, memory

def prepare_input(text, tokenizer, max_length):
    """Prepare the input following DeepSeek's format"""
    prompt = f"<|assistant|>{text}<|user|>"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    return inputs

def compute_similarity_loss(hidden_states1, hidden_states2, attention_mask=None):
    """Compute cosine similarity between hidden states"""
    last_hidden1 = hidden_states1[:, -1, :]
    last_hidden2 = hidden_states2[:, -1, :]
    
    last_hidden1 = F.normalize(last_hidden1, p=2, dim=1)
    last_hidden2 = F.normalize(last_hidden2, p=2, dim=1)
    
    similarity = torch.sum(last_hidden1 * last_hidden2, dim=1)
    similarity_loss = 1 - similarity.mean()
    
    return similarity_loss

def compute_levenshtein_distance(generated_text, target_text):
    """Compute the Levenshtein distance between generated and target text"""
    logger.info(f"Generated: {generated_text}")
    logger.info(f"Target: {target_text}")
    return levenshtein_distance(generated_text, target_text)

class StopOnTokens:
    def __init__(self, stop_ids_list):
        self.stop_ids_list = stop_ids_list

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        for stop_ids in self.stop_ids_list:
            if len(stop_ids) > len(input_ids[0]):
                continue
            if torch.equal(input_ids[0][-len(stop_ids):], torch.tensor(stop_ids, device=input_ids.device)):
                return True
        return False

def load_test_data(config):
    """Load and prepare test data based on config"""
    with open(config['data']['natural_language_file'], 'r') as f:
        natural_language = f.readlines()
    
    with open(config['data']['command_file'], 'r') as f:
        commands = f.readlines()
    
    test_data = {
        'natural_language': natural_language,
        'commands': commands
    }
    
    eval_size = config['data'].get('eval_size', 1.0)
    num_examples = int(len(test_data['natural_language']) * eval_size)
    test_data = {key: test_data[key][:num_examples] for key in test_data}
    
    logger.info(f"Using {eval_size*100}% of data ({num_examples} examples) for evaluation")
    
    return test_data 