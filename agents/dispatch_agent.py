from .base_agent import BaseAgent

class DispatchAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "dispatch_agent_prompt.json")

    def dispatch(self, task: str, summary: str, initial_plan: str) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        formatted_user_prompt = formatted_user_prompt.replace("{{summary}}", summary)
        formatted_user_prompt = formatted_user_prompt.replace("{{initial_plan}}", str(initial_plan))
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ]
        )
        return response
