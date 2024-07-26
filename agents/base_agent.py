import json
import os

class BaseAgent:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.load_prompt(prompt_file)

    def load_prompt(self, prompt_file):
        with open(os.path.join("prompts", prompt_file), "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        self.system_prompt = prompt_data["system_prompt"]
        self.user_prompt = prompt_data["user_prompt"]

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt)