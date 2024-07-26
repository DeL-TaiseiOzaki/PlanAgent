from .base_llm import BaseLLM
import torch

class NonAPIModels(BaseLLM):
    def __init__(self, model_path: str, use_vllm: bool = True):
        self.use_vllm = use_vllm
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_vllm:
            from vllm import LLM, SamplingParams
            self.model = LLM(model=model_path)
            self.sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, prompt: str) -> str:
        if self.use_vllm:
            outputs = self.model.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)