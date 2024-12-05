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
from datasets import Dataset

# At the top of the file, after the imports
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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
        
        # Get the appropriate device
        self.device = get_device()
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_and_prepare_data(self):
        """Load and prepare the dataset for training"""
        # Read both input and output files
        with open(self.data_path + "/cleaned_natural_language.txt", "r") as f_in, \
             open(self.data_path + "/cleaned_cmd.txt", "r") as f_out:
            inputs = f_in.readlines()
            outputs = f_out.readlines()
        
        # Take only 50% of the data
        # inputs = inputs[:len(inputs)//5]
        # outputs = outputs[:len(outputs)//5]
        
        # Create a dataset dictionary
        dataset_dict = {
            "train": {
                "text": [
                    f"Prompt: {input.strip()}\nResponse: {output.strip()}"
                    for input, output in zip(inputs, outputs)
                ]
            }
        }
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_dict(dataset_dict["train"])
        
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
            train_dataset=dataset,
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
        model_name="deepseek-ai/deepseek-coder-1.3b-base",
        data_path="data",
        output_dir="output/deepseek-ai_deepseek-coder-1_3b-base",
        max_length=512,
        batch_size=4,
        num_epochs=3,
        learning_rate=2e-5
    )
    
    finetuner.train()

if __name__ == "__main__":
    main()
