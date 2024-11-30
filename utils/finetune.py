from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import load_dataset
import os
from typing import Optional, Dict, Any

class ModelFinetuner:
    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        max_length: int = 512,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_and_prepare_data(self):
        """Load and prepare the dataset for training"""
        # Load dataset from the data directory
        dataset = load_dataset('text', data_files=self.data_path)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        return tokenized_dataset

    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            no_cuda=not torch.cuda.is_available(),
            push_to_hub=False,
        )

    def train(self):
        """Train the model"""
        # Prepare dataset
        dataset = self.load_and_prepare_data()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Initialize data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

def main():
    # Example usage
    finetuner = ModelFinetuner(
        model_name="facebook/opt-125m",
        data_path="data/*.txt",
        output_dir="output/finetuned-model",
        max_length=512,
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5
    )
    
    finetuner.train()

if __name__ == "__main__":
    main()
