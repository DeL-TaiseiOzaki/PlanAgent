from .base_agent import BaseAgent
from typing import Dict, List

class SummarizeTaskAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "summarize_task_agent_prompt.json")

    def summarize_task(self, task: str, retrieved_info: Dict, conversation_history: List[Dict]) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        formatted_user_prompt = formatted_user_prompt.replace("{{retrieved_info}}", str(retrieved_info))
        formatted_user_prompt = formatted_user_prompt.replace("{{conversation_history}}", str(conversation_history))
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ]
        )
        return self.parse_summarized_task(response)

    def parse_summarized_task(self, response: str) -> str:
        return response.strip()