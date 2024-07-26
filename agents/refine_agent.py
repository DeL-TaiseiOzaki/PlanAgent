from .base_agent import BaseAgent

class RefineAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "refine_agent_prompt.json")

    def refine_plan(self, subtask_id: str, max_step: int, modify_steps: int, max_plan_tree_depth: int, workspace_files: str, refine_node_message: str) -> str:
        formatted_user_prompt = self.user_prompt.replace("{{subtask_id}}", subtask_id)
        formatted_user_prompt = formatted_user_prompt.replace("{{max_step}}", str(max_step))
        formatted_user_prompt = formatted_user_prompt.replace("{{modify_steps}}", str(modify_steps))
        formatted_user_prompt = formatted_user_prompt.replace("{{max_plan_tree_depth}}", str(max_plan_tree_depth))
        formatted_user_prompt = formatted_user_prompt.replace("{{workspace_files}}", workspace_files)
        formatted_user_prompt = formatted_user_prompt.replace("{{refine_node_message}}", refine_node_message)
        response = self.generate(f"{self.system_prompt}\n\n{formatted_user_prompt}")
        return response