import anthropic
from .base_llm import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return message.content[0].text