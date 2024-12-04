import openai
import logging
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

def setup_openai():
    """Setup OpenAI API with key from environment"""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. GPT validation disabled.")
    return openai.api_key is not None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def validate_with_gpt4(generated_command: str, target_command: str, natural_language: str) -> dict:
    """
    Use GPT-4 to validate the generated command against the target command.
    
    Returns:
        dict: Contains validation results including correctness score and explanation
    """
    prompt = f"""You are an expert in bash commands. Please analyze these two commands and determine if they are functionally equivalent:

Natural Language Request: {natural_language}

Generated Command: {generated_command}
Target Command: {target_command}

Please provide your analysis in the following JSON format:
{{
    "functionally_equivalent": true/false,
    "correctness_score": <float between 0 and 1>,
    "explanation": "detailed explanation of the comparison",
    "key_differences": ["list of important differences if any"]
}}"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a bash command evaluation expert. Provide detailed analysis in the requested JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={ "type": "json_object" }
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        logger.error("Failed to parse GPT-4 response as JSON")
        return {
            "functionally_equivalent": False,
            "correctness_score": 0.0,
            "explanation": "Failed to parse GPT-4 response",
            "key_differences": ["Parse error"]
        } 