from .base_agent import BaseAgent
from typing import Dict, Any

class SummarizeTaskAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "summarize_task_agent_prompt.json")

    def summarize_task(self, task: str, conversation_history: list[Dict]) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        formatted_user_prompt = self.user_prompt.replace("{{conversation_history}}", str(conversation_history))
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ]
        )
        return response.strip()
