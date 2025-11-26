import requests
import re

def IDM(input_text: str, model_name: str = "llama2:13b") -> dict[str, str]:
    """    
    Args:
        input_text (str): User's raw question text
        
    Returns:
        Dict[str, str]: Contains 'intent' and 'query' keys
    """
    prompt = f"""As a professional question analyzer, follow these steps:

        1. Deeply understand: "{input_text}"
        2. Extract core intent (one concise sentence)
        3. Create search-optimized query (noun phrases/keywords)
        4. Format EXACTLY like:

        Intent: [single intent sentence]
        Query: [search terms]

        Example:
        User: How to prevent summer colds?
        Output:
        Intent: Identify effective prevention methods for seasonal summer colds
        Query: summer cold prevention effective methods

        Now analyze: "{input_text}"
        """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate", # "http://localhost:11434/api/generate"
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 80,
                }
            },
            timeout=80
        )
        response.raise_for_status()

        raw_response = response.json()["response"].strip()
        
        intent_pattern = r"Intent:\s*(.+?)(?=\nQuery:|\n\n|$)"
        query_pattern = r"Query:\s*(.+?)(?=\n|$)"
        
        intent = re.search(intent_pattern, raw_response, re.DOTALL)
        query = re.search(query_pattern, raw_response, re.DOTALL)

        if intent and query:
            # Sentence enforcement
            clean_intent = re.sub(r'[^a-zA-Z, \-].*', '', intent.group(1)).strip().rstrip('.')
            clean_query = re.sub(r'[^a-zA-Z, \-].*', '', query.group(1)).strip()
            
            return {
                "intent": clean_intent,
                "query": clean_query if len(clean_query) > 3 else input_text
            }

        return {"intent": "", "query": input_text}

    except Exception as e:
        print(f"Analysis Error: {str(e)}")
        return {"intent": "", "query": input_text}