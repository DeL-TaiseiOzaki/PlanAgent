import abc

class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass