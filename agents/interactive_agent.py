from typing import List, Dict, Tuple
from .base_agent import BaseAgent
import config

class InteractiveAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "interactive_agent_prompt.json")
        self.conversation_history = []
        self.max_questions = config.MAX_QUESTIONS

    def generate_question(self, task: str, missing_info: list[str]) -> str:
        prompt = f"{self.system_prompt}\n\nタスク: {task}\n\n不足情報:\n"
        for info in missing_info:
            prompt += f"- {info}\n"
        prompt += "\n会話履歴:\n"
        for entry in self.conversation_history:
            prompt += f"{entry['role']}: {entry['content']}\n"
        prompt += "\n次の質問を生成するか、タスクの実施計画が立てられるレベルの情報が得られたと判断した場合は'TASK_UNDERSTOOD'と応答してください:"
        return self.generate(prompt).strip()

    def interactive_information_collection(self, task: str, missing_info: list[str]) -> List[Dict]:
        print(f"タスク: {task}")
        print("タスクを完了するために必要な情報を収集します。")

        question_count = 0

        while question_count < self.max_questions:
            question = self.generate_question(task, missing_info)
            
            if question.upper() == "TASK_UNDERSTOOD":
                print("\n十分な情報が収集されました。タスクの理解が完了しました。")
                break

            print(f"\n質問: {question}")
            answer = input("あなたの回答: ")
            self.conversation_history.append({"role": "assistant", "content": question})
            self.conversation_history.append({"role": "user", "content": answer})
            
            question_count += 1

        if question_count >= self.max_questions:
            print("\n最大質問回数に達しました。収集された情報でタスクを進めます。")

        return self.conversation_history