import re
import requests

def IRM(question: str, materials: list[str], model_name: str = "llama2:13b") -> dict[str, str]:
    materials_str = "\n".join([f"{i+1}. {text}" for i, text in enumerate(materials)])
    
    prompt = f"""Analyze the following problem and retrieved materials step by step. 
First identify the gaps in current information, then generate both a refined intent and a search description for further investigation.

[Problem]: {question}
[Materials]:
{materials_str}

Let's think through this step by step:
1. First, analyze the user's original problem statement
2. Evaluate the relevance and completeness of retrieved materials
3. Identify specific aspects that need clarification or additional information
4. Determine the knowledge domains that should be explored next
5. Formulate a precise follow-up intent based on the identified gaps
6. Create a searchable description for targeted information retrieval

Generate the output strictly in this format:
New Intent: [Your refined intent statement here]
New Search Description: [Your search description here]"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", # "http://localhost:11434/api/generate"
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "max_tokens": 80,
                }
            }
        )
        response.raise_for_status()
        full_response = response.json()["response"]
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"intent": "", "description": ""}
    
    intent_pattern = r"New Intent:\s*(.*?)(?=\nNew Search Description:|\n|$)"
    search_pattern = r"New Search Description:\s*(.*?)(?=\n|$)"
    
    new_intent = re.search(intent_pattern, full_response, re.DOTALL)
    new_search = re.search(search_pattern, full_response, re.DOTALL)
    
    return {
        "intent": new_intent.group(1).strip() if new_intent else "",
        "description": new_search.group(1).strip() if new_search else ""
    }