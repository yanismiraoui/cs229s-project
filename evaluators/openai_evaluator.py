from openai import OpenAI
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict

logger = logging.getLogger(__name__)

def setup_openai():
    """Setup OpenAI API with key from file"""
    try:
        with open('OPENAI_API_KEY', 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            logger.warning("OPENAI_API_KEY file is empty. GPT validation disabled.")
            return False
        global client
        client = OpenAI(api_key=api_key)
        return True
    except FileNotFoundError:
        logger.warning("OPENAI_API_KEY file not found. GPT validation disabled.")
        return False
    except Exception as e:
        logger.warning(f"Failed to read OPENAI_API_KEY file: {str(e)}. GPT validation disabled.")
        return False

def clean_json_response(response_text: str) -> str:
    """Clean JSON response by removing markdown code blocks and other formatting"""
    # Remove markdown code block syntax if present
    response_text = response_text.replace('```json', '').replace('```', '')
    # Strip whitespace
    response_text = response_text.strip()
    return response_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def batch_validate_with_gpt4(
    generated_commands: List[str],
    target_commands: List[str],
    natural_language_requests: List[str],
    batch_size: int = 10
) -> List[Dict]:
    """
    Use GPT-4 to validate multiple commands in batches.
    
    Args:
        generated_commands: List of generated commands
        target_commands: List of target commands
        natural_language_requests: List of natural language requests
        batch_size: Number of commands to validate in each batch
        
    Returns:
        List[Dict]: List of validation results
    """
    results = []
    
    for i in range(0, len(generated_commands), batch_size):
        batch_gen = generated_commands[i:i + batch_size]
        batch_target = target_commands[i:i + batch_size]
        batch_nl = natural_language_requests[i:i + batch_size]
        
        # Create a batch prompt
        batch_prompt = "Analyze these bash commands and determine if they are functionally equivalent. Respond with ONLY a JSON array - no markdown formatting, no explanations:\n\n"
        for j, (gen, target, nl) in enumerate(zip(batch_gen, batch_target, batch_nl)):
            batch_prompt += f"Example {j+1}:\n"
            batch_prompt += f"Natural Language: {nl}\n"
            batch_prompt += f"Generated: {gen}\n"
            batch_prompt += f"Target: {target}\n\n"
        
        batch_prompt += """Return ONLY a JSON array in this format (no markdown, no other text):
[
    {
        "example_id": 1,
        "functionally_equivalent": true/false,
        "correctness_score": <float between 0 and 1>
    },
    ...
]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for faster inference
                messages=[
                    {"role": "system", "content": "You are a bash command evaluation expert. Return ONLY JSON without any markdown formatting or additional text."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0
            )
            
            response_text = clean_json_response(response.choices[0].message.content)
            batch_results = json.loads(response_text)
            results.extend(batch_results)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT-4 response as JSON: {str(e)}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            # Add failure results for this batch
            for _ in range(len(batch_gen)):
                results.append({
                    "functionally_equivalent": False,
                    "correctness_score": 0.0
                })
        except Exception as e:
            logger.error(f"GPT-4 validation failed: {str(e)}")
            # Add failure results for this batch
            for _ in range(len(batch_gen)):
                results.append({
                    "functionally_equivalent": False,
                    "correctness_score": 0.0
                })
    
    return results