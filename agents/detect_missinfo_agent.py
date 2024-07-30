from .base_agent import BaseAgent

class DetectMissinfoAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "detect_missinfo_agent_prompt.json")

    def detect_missing_info(self, task: str) -> list[str]:
        formatted_user_prompt = self.user_prompt.replace("{{task}}", task)
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}")
        return self.parse_missing_info(response)

    def parse_missing_info(self, response: str) -> list[str]:
        return [line.strip() for line in response.split('\n') if line.strip()]
