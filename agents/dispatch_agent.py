from .base_agent import BaseAgent

class DispatchAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "dispatch_agent_prompt.json")

    def dispatch(self, task: str, example_system_prompt: str, example_user_prompt: str) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{example_system_prompt}}", example_system_prompt)
        formatted_user_prompt = formatted_user_prompt.replace("{{example_user_prompt}}", example_user_prompt)
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}\n\nTask: {task}")
        return response