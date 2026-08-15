from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os 
from typing import List, Optional, Dict

class ModelAI:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in the environment")
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = "gpt-5"

    def chat(
        self, 
        messages: List[Dict],
        system_prompt: str,
        model: Optional[str] = None,
        max_tokens=5000,
    ):
        try:
            full_message = [
                {
                    "role": "system", "content": system_prompt
                }
            ] + messages
            response = self.client.chat.completions.create(
                model = self.default_model or model,
                max_completion_tokens=max_tokens,
                messages = full_message
            )
            reply = response.choices[0].message.content  
            usage = response.usage  
            return usage, reply

        except Exception as e:
            return "error", str(e)