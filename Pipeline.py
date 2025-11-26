from utils.IDM import IDM
from utils.IRM import IRM
from utils.SJM import SJM
from retriever_HT import retrieve_relevant_passages_hotpotqa

import requests
from typing import List, Dict
import json

class RAGPipeline:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):# http://localhost:11434/api/generate
        self.ollama_url = ollama_url
        
    def generate_answer(self, question: str, current_data, model_name: str = "llama2:13b") -> str:

        analysis_chain = []
        iteration = 1
        
        analysis = IDM(question, model_name)
        current_intent = analysis['intent']
        current_query = analysis['query']
        
        sufficient = False
        
        while iteration <= 3 and not sufficient:
            passages = retrieve_relevant_passages_hotpotqa(current_query, current_data, k=5)

            analysis_chain.append({
                "iteration": iteration,
                "intent": current_intent,
                "query": current_query,
                "passages": passages
            })
            
            sufficient = SJM(current_intent, passages, model_name)
            
            if not sufficient and iteration < 3:
                new_params = IRM(current_intent, passages, model_name)
                current_intent = new_params['intent']
                current_query = new_params['description']
            
            iteration += 1
        
        prompt = self._build_chain_prompt(question, analysis_chain)
        # print(analysis_chain)
        return self._call_llm(prompt, model_name)
    
    def _build_chain_prompt(self, question: str, analysis_chain: List[Dict]) -> str:
        prompt = f"Original question:{question}\n\nAnalysis process:"
        for step in analysis_chain:
            prompt += f"\n\nstep {step['iteration']}:"
            prompt += f"\n- intent:{step['intent']}"
            prompt += f"\n- retrieve passages:{', '.join(step['passages'])}..."
        prompt += "\n\nNow, refer the analysis process, directly give the answer to the original question concisely, without any explanations or extra description"
        return prompt
    
    def _call_llm(self, prompt: str, model_name: str = "llama2:13b") -> str:
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                    }
                }
            )
            # print(response.json()["response"])
            return response.json()["response"]
        except Exception as e:
            return f"Error:{str(e)}"