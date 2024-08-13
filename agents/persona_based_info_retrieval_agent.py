from .base_agent import BaseAgent
from datasets import load_dataset
import config
import logging
import os
import json

class PersonaBasedInfoRetrievalAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "persona_based_info_retrieval_agent_prompt.json")
        self.persona_data = None

    def _load_persona_data(self):
        if self.persona_data is None:
            try:
                with open(config.PERSONA_FILE_PATH, 'r', encoding='utf-8') as f:
                    self.persona_data = json.load(f)
                logging.info(f"Persona data loaded from {config.PERSONA_FILE_PATH}")
            except FileNotFoundError:
                logging.error(f"Persona file not found at {config.PERSONA_FILE_PATH}")
                raise
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in persona file at {config.PERSONA_FILE_PATH}")
                raise

    def retrieve_info(self, missing_info: list[str]) -> dict:
        self._load_persona_data()
        user_id = config.CURRENT_USER_ID
        try:
            persona, chat_history = self._get_user_data(user_id)
            formatted_user_prompt = self.user_prompt.replace("{{persona}}", str(persona))
            formatted_user_prompt = formatted_user_prompt.replace("{{chat_history}}", str(chat_history))
            formatted_user_prompt = formatted_user_prompt.replace("{{missing_info}}", str(missing_info))
            
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ]
            )
            return self.parse_retrieved_info(response)
        except KeyError as e:
            logging.error(f"Error retrieving user data: {e}")
            return {}

    def _get_user_data(self, user_id: int) -> tuple[dict, list]:
        user_data = next((item for item in self.persona_data if item['ID'] == user_id), None)
        if not user_data:
            logging.error(f"User with ID {user_id} not found in the persona data")
            raise ValueError(f"User with ID {user_id} not found in the dataset")
        
        persona = {
            'Name': user_data['Name'],
            'Age': user_data['Age'],
            'Work': user_data['Work'],
            'Character': user_data['Character'],
            'SoV': user_data['SoV']
        }
        chat_history = user_data['history']
        return persona, chat_history

    def parse_retrieved_info(self, response: str) -> dict:
        return response