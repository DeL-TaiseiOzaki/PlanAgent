from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from .base_llm import BaseLLM
from config import ANTHROPIC_MODEL

class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, temperature: float, max_tokens: int):
        self.client = Anthropic(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.completion(
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            model=ANTHROPIC_MODEL,
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature
        )
        return response.completion