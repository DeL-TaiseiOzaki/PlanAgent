from .base_agent import BaseAgent
import config

class InteractiveAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "interactive_agent_prompt.json")
        self.conversation_history = []
        self.max_questions = config.MAX_QUESTIONS
        self.question_count = 0
        self.task_understood = False

    def generate(self, user_input: str) -> str:
        if self.task_understood:
            return "タスクの理解が完了しました。これ以上の質問は必要ありません。"

        if self.question_count >= self.max_questions:
            self.task_understood = True
            return "最大質問回数に達しました。これまでに収集された情報でタスクを進めます。"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]
        
        response = self.llm.generate(messages)
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        self.question_count += 1
        
        if response.upper() == "TASK_UNDERSTOOD":
            self.task_understood = True
            return "タスクの理解が完了しました。これ以上の質問は必要ありません。"
        
        return response

    def reset(self):
        self.conversation_history = []
        self.question_count = 0
        self.task_understood = False

    def interactive_information_collection(self, task: str, missing_info: list[str]) -> list[dict]:
        self.reset()  # 新しいタスクのために状態をリセット
        print(f"タスク: {task}")
        print("タスクを完了するために必要な情報を収集します。")

        while not self.task_understood and self.question_count < self.max_questions:
            prompt = f"タスク: {task}\n\n不足情報:\n"
            for info in missing_info:
                prompt += f"- {info}\n"
            prompt += "\n次の質問を生成するか、タスクの実施計画が立てられるレベルの情報が得られたと判断した場合は'TASK_UNDERSTOOD'と応答してください。"

            question = self.generate(prompt)
            
            if self.task_understood:
                print("\n十分な情報が収集されました。タスクの理解が完了しました。")
                break

            print(f"\n質問: {question}")
            answer = input("あなたの回答: ")
            
            if answer.upper() == "TASK_UNDERSTOOD":
                self.task_understood = True
                print("\n十分な情報が収集されました。タスクの理解が完了しました。")
                break

        if self.question_count >= self.max_questions:
            print("\n最大質問回数に達しました。収集された情報でタスクを進めます。")

        return self.conversation_history