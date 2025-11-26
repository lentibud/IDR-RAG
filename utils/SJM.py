import requests
import re

def SJM(intent: str, retrieved_docs: list[str], model_name: str = "llama2:13b") -> bool:
    """
    Determine if the retrieved results meet the user's intent requirements.
    
    Parameters:
        intent: User intent analysis text (str)
        retrieved_docs: List of retrieved documents (list[str])
        
    Returns:
        bool: Judgment result of whether the information is sufficient
    """
    prompt = f"""You are an analytical assistant specializing in information adequacy assessment. Follow this thinking process:

    [User Intent Analysis]
    {intent}

    [Retrieved Documents]
    {chr(10).join(f'- {doc}' for doc in retrieved_docs)}

    [Analysis Steps]
    1. Identify key requirements from the user intent
    2. Map each requirement to relevant information in the documents
    3. Detect any missing elements or ambiguities
    4. Consider potential follow-up questions needed
    5. Final adequacy conclusion (respond EXACTLY in this format):
       Final Answer: [YES/NO]"""

    try:
        # Call local Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",# "http://localhost:11434/api/generate"
            json={
                "model": model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "max_tokens": 80,
                    "stop": ["\nFinal Answer:"]
                }
            },
            timeout=200
        )
        response.raise_for_status()

        # Parse model response
        full_response = response.json()["response"].strip()
        # print(full_response)
        return parse_final_answer(full_response)

    except requests.exceptions.RequestException as e:
        print(f"API call error: {e}")
        return False
    except KeyError:
        print("Abnormal response format")
        return False

def parse_final_answer(response_text: str) -> bool:
    """Enhanced parser: supports multiple format variants"""
    patterns = [
        # Handle JSON-like output
        r'"Final Answer"\s*:\s*["\']*(YES|NO)["\']*',
        # Handle standard CoT format
        r'Final Answer:\s*(YES|NO)\b',
        # Handle variants with punctuation
        r'Final Answer\s*[—-]\s*(YES|NO)\b',
        # Handle code block format
        r'```(?:json)?\s*{\s*"Final Answer"\s*:\s*"(YES|NO)"\s*}'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper() == "YES"
    
    # Safe fallback: keyword search
    yes_flags = ['sufficient', 'adequate', 'complete', 'yes']
    no_flags = ['insufficient', 'inadequate', 'missing', 'no']
    
    if any(flag in response_text.lower() for flag in yes_flags):
        return True
    if any(flag in response_text.lower() for flag in no_flags):
        return False
    
    return False