import subprocess
import tempfile
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def validate_bash_syntax(command: str) -> Tuple[bool, str]:
    """
    Validate bash command syntax without executing it.
    
    Args:
        command: The bash command to validate
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Create a temporary file to store the command
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as tmp:
        # Write the command to the temp file
        tmp.write(command)
        tmp.flush()
        
        try:
            # Run bash -n for syntax checking only
            result = subprocess.run(
                ['bash', '-n', tmp.name],
                capture_output=True,
                text=True
            )
            
            # Check if syntax is valid
            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr.strip()
                
        except subprocess.SubprocessError as e:
            return False, f"Failed to run syntax check: {str(e)}"

def batch_validate_commands(commands: list[str]) -> list[dict]:
    """
    Validate a batch of bash commands.
    
    Args:
        commands: List of bash commands to validate
    
    Returns:
        list[dict]: List of validation results with format:
            {
                'command': str,
                'is_valid': bool,
                'error': str
            }
    """
    results = []
    for cmd in commands:
        is_valid, error = validate_bash_syntax(cmd)
        results.append({
            'command': cmd,
            'is_valid': is_valid,
            'error': error
        })
    return results

# Example usage
if __name__ == "__main__":
    test_commands = [
        "ls -la",  # Valid
        "echo 'hello world'",  # Valid
        "for i in {1..5}; do echo $i; done",  # Valid
        "if [ -f file.txt ]; then cat file.txt; fi",  # Valid
        "ls -la |",  # Invalid - incomplete pipe
        "for i in {1..5}; do echo $i;"  # Invalid - unclosed loop
    ]
    
    results = batch_validate_commands(test_commands)
    
    print("\nValidation Results:")
    print("==================")
    for result in results:
        status = "✓" if result['is_valid'] else "✗"
        print(f"\n{status} Command: {result['command']}")
        if not result['is_valid']:
            print(f"   Error: {result['error']}") 