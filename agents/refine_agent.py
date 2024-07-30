from typing import List, Dict
from .base_agent import BaseAgent

class RefineAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "refine_agent_prompt.json")

    def refine_plan(self, task: str, max_step: int, modify_steps: int, max_plan_tree_depth: int, summary: str, conversation_history: List[Dict]) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        formatted_user_prompt = formatted_user_prompt.replace("{{max_step}}", str(max_step))
        formatted_user_prompt = formatted_user_prompt.replace("{{modify_steps}}", str(modify_steps))
        formatted_user_prompt = formatted_user_prompt.replace("{{max_plan_tree_depth}}", str(max_plan_tree_depth))
        formatted_user_prompt = formatted_user_prompt.replace("{{summary}}", summary)
        formatted_user_prompt = formatted_user_prompt.replace("{{conversation_history}}", str(conversation_history))
        
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}")
        return response.strip()