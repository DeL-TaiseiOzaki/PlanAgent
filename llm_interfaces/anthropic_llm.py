from anthropic import Anthropic
from .base_llm import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict] or str) -> str:
        if isinstance(messages, str):
            # If a single string is provided, treat it as a user message
            messages = [{"role": "user", "content": messages}]

        # Extract system message and user messages
        system_message = None
        user_messages = []

        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
            else:
                user_messages.append(message)

        # If no system message is provided, use a default one
        if not system_message:
            system_message = "You are a helpful AI assistant."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=user_messages
        )
        return response.content[0].text