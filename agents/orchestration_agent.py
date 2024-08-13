from .base_agent import BaseAgent
from .ToolAgents.web_search_agent import WebSearchAgent

class OrchestrationAgent(BaseAgent):
    def __init__(self, llm, web_search_api_key):
        super().__init__(llm, "orchestration_agent_prompt.json")
        self.web_search_agent = WebSearchAgent(llm, web_search_api_key)

    def execute_tasks(self, decomposed_tasks):
        results = []
        for task in decomposed_tasks:
            task_result = self.execute_single_task(task)
            results.append(task_result)
        return results

    def execute_single_task(self, task):
        if "search" in task['name'].lower():
            search_results = self.web_search_agent.search(task['input'])
            return {"task": task['name'], "result": search_results}
        else:
            # Web検索以外のタスクの場合、LLMを使用して処理
            prompt = self.format_task_prompt(task)
            response = self.generate(prompt)
            return {"task": task['name'], "result": response}

    def format_task_prompt(self, task):
        return f"Execute the following task:\nName: {task['name']}\nDescription: {task['description']}\nInput: {task['input']}\nExpected Output: {task['output']}"
