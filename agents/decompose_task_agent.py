from .base_agent import BaseAgent
from typing import List, Dict

class DecomposeTaskAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "decompose_task_agent_prompt.json")

    def decompose_task(self, concrete_task: str) -> List[Dict[str, str]]:
        formatted_user_prompt = self.user_prompt.replace("{{concrete_task}}", concrete_task)
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ]
        )
        return self.parse_decomposed_tasks(response)

    def parse_decomposed_tasks(self, response: str) -> List[Dict[str, str]]:
        """
        decomposed_tasks = []
        current_task = {}
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("タスク"):
                if current_task:
                    decomposed_tasks.append(current_task)
                current_task = {"name": line.split(":", 1)[1].strip()}
            elif line.startswith("説明:"):
                current_task["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("入力:"):
                current_task["input"] = line.split(":", 1)[1].strip()
            elif line.startswith("出力:"):
                current_task["output"] = line.split(":", 1)[1].strip()
        if current_task:
            decomposed_tasks.append(current_task)
            """
        return response