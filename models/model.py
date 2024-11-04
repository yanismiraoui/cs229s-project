from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Dict, Optional, Union, List
import torch

class BoltModel:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cpu")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["base_model"],
            padding_side="left",
            truncation_side="left"
        )
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Apply optimizations
        self.apply_optimizations()
    
    def apply_optimizations(self):
        """Apply various optimization techniques"""
        if self.config["model"]["quantization"]["enabled"]:
            # Note: Implement quantization logic here
            pass
            
        if self.config["model"]["pruning"]["enabled"]:
            # Note: Implement pruning logic here
            pass
    
    def prepare_input(self, text: Union[str, List[str]]) -> dict:
        """Prepare input for the model"""
        prompt_template = "Convert this text to a bash command: {}\n\nBash command:"
        
        if isinstance(text, str):
            text = [text]
            
        prompts = [prompt_template.format(t) for t in text]
        
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=self.config["model"]["max_length"],
            truncation=True
        ).to(self.device)
    
    def generate_command(
        self, 
        natural_language: Union[str, List[str]], 
        **gen_kwargs
    ) -> Union[str, List[str]]:
        """Convert natural language to bash command"""
        inputs = self.prepare_input(natural_language)
        
        # Set default generation parameters
        generation_config = {
            "max_length": self.config["model"]["max_length"],
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            **gen_kwargs  # Allow overriding defaults
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode outputs
        commands = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Extract just the command part (after the prompt)
        commands = [cmd.split("Bash command:")[-1].strip() for cmd in commands]
        
        return commands[0] if isinstance(natural_language, str) else commands
    
    def get_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Return the underlying model and tokenizer"""
        return self.model, self.tokenizer