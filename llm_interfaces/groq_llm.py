import groq
from .base_llm import BaseLLM

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content