from .base_llm import BaseLLM
import torch

class NonAPIModels(BaseLLM):
    def __init__(self, model_path: str, use_vllm: bool = True, temperature: float = 0.7, max_tokens: int = 1000):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if use_vllm:
            from vllm import LLM, SamplingParams
            self.model = LLM(model=model_path)
            self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, prompt: str) -> str:
        if self.use_vllm:
            outputs = self.model.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)