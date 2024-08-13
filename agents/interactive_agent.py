from .base_agent import BaseAgent
import config
from typing import List, Union
from typing import List, Tuple
import json

class InteractiveAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "interactive_agent_prompt.json")
        self.conversation_history = []
        self.max_questions = config.MAX_QUESTIONS
        self.question_count = 0
        self.task_understood = False

    def interactive_information_collection(self, task: str, missing_info: str, retrieved_info: str) -> Tuple[List[dict], str]:
        self.reset()  # 新しいタスクのために状態をリセット
        while not self.task_understood and self.question_count < self.max_questions:
            prompt = self._generate_prompt(task, missing_info, retrieved_info)
            response = self.generate(prompt)
            
            if response.upper() == "TASK_UNDERSTOOD":
                self.task_understood = True
                break

            print(f"\n質問: {response}")
            answer = input("あなたの回答: ")

            self.conversation_history.append({"role": "assistant", "content": response})
            self.conversation_history.append({"role": "user", "content": answer})
            self.question_count += 1

            # 各回答後に不足情報と取得済み情報を更新
            prompt_for_update = self._generate_update_prompt(task, missing_info, retrieved_info, response, answer)
            update_response = self.generate(prompt_for_update)
            missing_info, retrieved_info = self._parse_update_response(update_response)

        return self.conversation_history, retrieved_info

    def _generate_prompt(self, task: str, missing_info: str, retrieved_info: str) -> str:
        prompt = f"タスク: {task}\n\n"
        prompt += f"取得済み情報:\n{retrieved_info}\n\n"
        prompt += f"不足情報:\n{missing_info}\n\n"
        prompt += "次の質問を生成するか、タスクの実施計画が立てられるレベルの情報が得られたと判断した場合は'TASK_UNDERSTOOD'と応答してください。"
        return prompt

    def _generate_update_prompt(self, task: str, missing_info: str, retrieved_info: str, last_question: str, last_answer: str) -> str:
        prompt = f"タスク: {task}\n\n"
        prompt += f"現在の取得済み情報:\n{retrieved_info}\n\n"
        prompt += f"現在の不足情報:\n{missing_info}\n\n"
        prompt += f"直前の質問: {last_question}\n"
        prompt += f"ユーザーの回答: {last_answer}\n\n"
        prompt += "上記の情報を基に、取得済み情報と不足情報を更新してください。"
        prompt += "更新後の情報を以下のフォーマットで出力してください：\n"
        prompt += "取得済み情報:\n(更新された取得済み情報を文章形式で)\n"
        prompt += "不足情報:\n(更新された不足情報を文章形式で)\n"
        return prompt

    def _parse_update_response(self, response: str) -> Tuple[str, str]:
        lines = response.split('\n')
        retrieved_info = ""
        missing_info = ""
        current_section = None

        for line in lines:
            if line.strip() == "取得済み情報:":
                current_section = "retrieved"
            elif line.strip() == "不足情報:":
                current_section = "missing"
            elif current_section == "retrieved":
                retrieved_info += line + "\n"
            elif current_section == "missing":
                missing_info += line + "\n"

        return missing_info.strip(), retrieved_info.strip()

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.llm.generate(messages)

    def reset(self):
        self.conversation_history = []
        self.question_count = 0
        self.task_understood = False