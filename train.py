import yaml
from utils.preprocessing import load_data_files, prepare_dataset, split_dataset
from models.model import BoltModel
from models.trainer import BoltTrainer

def main():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    bolt_model = BoltModel(config)
    model, tokenizer = bolt_model.get_model_and_tokenizer()
    
    # Load and prepare data
    dataset = load_data_files(
        config["data"]["natural_language_file"],
        config["data"]["command_file"]
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_size=config["data"]["train_size"],
        val_size=config["data"]["val_size"]
    )
    
    # Prepare datasets for training
    train_dataset = prepare_dataset(train_dataset, tokenizer, config)
    val_dataset = prepare_dataset(val_dataset, tokenizer, config)
    test_dataset = prepare_dataset(test_dataset, tokenizer, config)
    
    # Initialize trainer
    trainer = BoltTrainer(model, tokenizer, config)
    
    # Train the model
    trainer.train(train_dataset, val_dataset)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

if __name__ == "__main__":
    main() 