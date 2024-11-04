import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from typing import Dict, Optional
from datasets import Dataset

class BoltTrainer:
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Create training arguments
        self.training_args = TrainingArguments(
            output_dir="./bolt_results",
            num_train_epochs=config["training"]["num_epochs"],
            per_device_train_batch_size=config["training"]["batch_size"],
            per_device_eval_batch_size=config["training"]["batch_size"],
            warmup_steps=config["training"]["warmup_steps"],
            learning_rate=float(config["training"]["learning_rate"]),
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            load_best_model_at_end=False
        )
        
        # Create data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            max_length=config["model"]["max_length"]
        )
        
        # Initialize the Trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            data_collator=self.data_collator,
            tokenizer=tokenizer,
            train_dataset=None,
            eval_dataset=None
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Train the model using the Hugging Face Trainer"""
        self.trainer.train_dataset = train_dataset
        self.trainer.eval_dataset = eval_dataset
        return self.trainer.train()
    
    def evaluate(self, eval_dataset: Dataset):
        """Evaluate the model using the Hugging Face Trainer"""
        return self.trainer.evaluate(eval_dataset) 