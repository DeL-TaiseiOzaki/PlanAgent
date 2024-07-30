from .base_agent import BaseAgent
from typing import Dict, Any

class SummarizeTaskAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "summarize_task_agent_prompt.json")

    def summarize_task(self, task: str, conversation_history: list[Dict]) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        formatted_user_prompt = formatted_user_prompt.replace("{{conversation_history}}", str(conversation_history))
        
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}")
        return response.strip()