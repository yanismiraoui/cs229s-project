from datasets import Dataset
from typing import Dict, List, Tuple
import pandas as pd

def load_data_files(nl_file: str, cmd_file: str) -> Dataset:
    """Load and preprocess the data files into a Hugging Face Dataset"""
    
    # Read the files
    with open(nl_file, 'r', encoding='utf-8') as f:
        natural_language = [line.strip() for line in f.readlines()]
    
    with open(cmd_file, 'r', encoding='utf-8') as f:
        commands = [line.strip() for line in f.readlines()]
    
    # Create a dictionary for the Dataset
    data_dict = {
        'natural_language': natural_language,
        'command': commands
    }
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data_dict)
    
    return dataset

def prepare_dataset(dataset: Dataset, tokenizer, config: Dict) -> Dataset:
    """Prepare dataset for training by tokenizing inputs and targets"""
    
    def tokenize_function(examples):
        # Prepare input text
        model_inputs = tokenizer(
            ["Convert this text to a bash command: " + nl for nl in examples["natural_language"]],
            max_length=config["data"]["max_source_length"],
            padding="max_length",
            truncation=True
        )
        
        # Prepare target commands
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["command"],
                max_length=config["data"]["max_target_length"],
                padding="max_length",
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def split_dataset(dataset: Dataset, train_size: float = 0.8, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train, validation, and test sets"""
    
    # Calculate split sizes
    test_size = 1.0 - train_size - val_size
    
    # Split dataset
    splits = dataset.train_test_split(
        train_size=train_size,
        test_size=(val_size + test_size),
        shuffle=True,
        seed=42
    )
    
    train_dataset = splits["train"]
    
    # Further split the test portion into validation and test
    remaining_splits = splits["test"].train_test_split(
        train_size=val_size/(val_size + test_size),
        test_size=test_size/(val_size + test_size),
        shuffle=True,
        seed=42
    )
    
    return train_dataset, remaining_splits["train"], remaining_splits["test"] 