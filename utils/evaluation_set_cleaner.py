from together import Together
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_together_ai():
    """Setup TogetherAI with API key from config"""
    config = load_config()
    client = Together(api_key=config["together_ai"]["api_key"])
    return client

def load_evaluation_sets():
    """Load the original command and natural language instruction files"""
    cmd_path = Path("data/cmd.txt")
    nl_path = Path("data/natural_language.txt")
    
    with open(cmd_path, "r") as f:
        commands = f.readlines()
    with open(nl_path, "r") as f:
        instructions = f.readlines()
    
    # Strip whitespace and filter out empty lines
    commands = [cmd.strip() for cmd in commands if cmd.strip()]
    instructions = [instr.strip() for instr in instructions if instr.strip()]
    
    return commands, instructions

def is_command_valid(client, command, instruction):
    """Use TogetherAI to check if a command is valid and runnable without environment setup"""
    prompt = f"""You are validating Unix commands. Determine if the given command can be run on a basic Unix system 
WITHOUT requiring any:
- package installation
- environment setup
- configuration
- sudo privileges
- external dependencies
- file creation/existence
- network access

The command should work with just basic Unix utilities that come pre-installed.

Only respond with 'VALID' or 'INVALID'.

Natural Language Description: {instruction}
Command: {command}

Can this command be run on a basic Unix system without any setup?"""

    response = client.chat.completions.create(
        messages=[
                {"role": "system", "content": prompt}
            ],
        stream=False,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        max_tokens=5,
        temperature=0.1,
    )
    result = response.choices[0].message.content
    return result == "VALID"

def clean_evaluation_set():
    """Clean the evaluation set and save valid commands to new files"""
    client = setup_together_ai()
    commands, instructions = load_evaluation_sets()
    
    if len(commands) != len(instructions):
        raise ValueError("Number of commands and instructions don't match")
    
    # Create list of indices and shuffle them
    indices = list(range(len(commands)))
    random.shuffle(indices)
    
    cleaned_commands = []
    cleaned_instructions = []
    target_count = 200
    
    logger.info(f"Processing randomly selected commands until {target_count} valid pairs are found...")
    
    for idx in tqdm(indices):
        cmd = commands[idx]
        instr = instructions[idx]
        try:
            if is_command_valid(client, cmd, instr):
                cleaned_commands.append(cmd)
                cleaned_instructions.append(instr)
                if len(cleaned_commands) >= target_count:
                    logger.info(f"Reached target of {target_count} valid commands. Stopping...")
                    break
            else:
                logger.info(f"Invalid command: {cmd}")
        except Exception as e:
            logger.error(f"Error processing command '{cmd}': {str(e)}")
    
    # Save cleaned data
    output_cmd_path = Path("data/cleaned_cmd.txt")
    output_nl_path = Path("data/cleaned_natural_language.txt")
    
    with open(output_cmd_path, "w") as f:
        f.write("\n".join(cleaned_commands))
    
    with open(output_nl_path, "w") as f:
        f.write("\n".join(cleaned_instructions))
    
    logger.info(f"Cleaned evaluation set saved. {len(cleaned_commands)} valid commands found.")

if __name__ == "__main__":
    clean_evaluation_set()

