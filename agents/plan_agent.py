from typing import List
from .base_agent import BaseAgent

class PlanAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "plan_agent_prompt.json")

    def initial_plan_generation(self, task: str) -> List[str]:
        formatted_user_prompt = self.user_prompt.replace("{{query}}", task)
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}")
        return self.parse_subtasks(response)

    def parse_subtasks(self, response: str) -> List[str]:
        return [line.strip() for line in response.split('\n') if line.strip()]