import sys
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Callable
from llm_interfaces.base_llm import BaseLLM
from agents.detect_missinfo_agent import DetectMissinfoAgent
from agents.persona_based_info_retrieval_agent import PersonaBasedInfoRetrievalAgent
from agents.interactive_agent import InteractiveAgent
from agents.summarize_task_agent import SummarizeTaskAgent
from agents.decompose_task_agent import DecomposeTaskAgent
import config
import logging

def get_llm(llm_type: str, temperature: float, max_tokens: int) -> BaseLLM:
    if llm_type == "openai":
        from llm_interfaces.openai_llm import OpenAILLM
        return OpenAILLM(config.OPENAI_API_KEY, config.OPENAI_MODEL, temperature, max_tokens)
    elif llm_type == "anthropic":
        from llm_interfaces.anthropic_llm import AnthropicLLM
        return AnthropicLLM(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL, temperature, max_tokens)
    elif llm_type == "groq":
        from llm_interfaces.groq_llm import GroqLLM
        return GroqLLM(config.GROQ_API_KEY, config.GROQ_MODEL, temperature, max_tokens)
    elif llm_type == "non_api":
        from llm_interfaces.non_api_models import NonAPIModels
        return NonAPIModels(config.NON_API_MODEL_PATH, config.USE_VLLM, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

def main(task: str, 
         detect_llm_type: str, detect_temperature: float, detect_max_tokens: int,
         persona_llm_type: str, persona_temperature: float, persona_max_tokens: int,
         interactive_llm_type: str, interactive_temperature: float, interactive_max_tokens: int,
         summarize_llm_type: str, summarize_temperature: float, summarize_max_tokens: int,
         decompose_llm_type: str, decompose_temperature: float, decompose_max_tokens: int,
         output_dir: str,
         log_callback: Callable[[str], None] = print) -> None:

    logger = logging.getLogger("AgentSystemLogger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    def log_callback(message):
        logger.info(message)

    log_callback("エージェントシステムを初期化中...")

    log_callback("LLMを初期化中...")
    detect_llm = get_llm(detect_llm_type, detect_temperature, detect_max_tokens)
    persona_llm = get_llm(persona_llm_type, persona_temperature, persona_max_tokens)
    interactive_llm = get_llm(interactive_llm_type, interactive_temperature, interactive_max_tokens)
    summarize_llm = get_llm(summarize_llm_type, summarize_temperature, summarize_max_tokens)
    decompose_llm = get_llm(decompose_llm_type, decompose_temperature, decompose_max_tokens)

    output = {
        "task": task,
        "agents": {}
    }

    # DetectMissinfoAgent
    log_callback("不足情報の検出を開始...")
    detect_missinfo_agent = DetectMissinfoAgent(detect_llm)
    missing_info = detect_missinfo_agent.detect_missing_info(task)
    log_callback(f"検出された不足情報: {missing_info}")
    output["agents"]["DetectMissinfoAgent"] = {
        "input": task,
        "output": missing_info
    }

    # PersonaBasedInfoRetrievalAgent
    log_callback("ペルソナベースの情報検索を開始...")
    persona_agent = PersonaBasedInfoRetrievalAgent(persona_llm)
    retrieved_info = persona_agent.retrieve_info(missing_info)
    log_callback("ペルソナベースの情報検索が完了しました")
    output["agents"]["PersonaBasedInfoRetrievalAgent"] = {
        "input": missing_info,
        "output": retrieved_info
    }

    # InteractiveAgent
    log_callback("対話的情報収集を開始...")
    interactive_agent = InteractiveAgent(interactive_llm)
    conversation_history, updated_retrieved_info = interactive_agent.interactive_information_collection(task, missing_info, retrieved_info)
    log_callback("対話的情報収集が完了しました")
    output["agents"]["InteractiveAgent"] = {
        "input": {
            "task": task,
            "missing_info": missing_info,
            "retrieved_info": retrieved_info
        },
        "output": {
            "conversation_history": conversation_history,
            "updated_retrieved_info": updated_retrieved_info
        }
    }

    # SummarizeTaskAgent
    log_callback("タスクの具体化を開始...")
    summarize_task_agent = SummarizeTaskAgent(summarize_llm)
    concrete_task = summarize_task_agent.summarize_task(task, updated_retrieved_info, conversation_history)
    log_callback("タスクの具体化が完了しました")
    output["agents"]["SummarizeTaskAgent"] = {
        "input": {
            "task": task,
            "retrieved_info": updated_retrieved_info,
            "conversation_history": conversation_history
        },
        "output": concrete_task
    }

    # DecomposeTaskAgent
    log_callback("タスクの分解を開始...")
    decompose_agent = DecomposeTaskAgent(decompose_llm)
    decomposed_tasks = decompose_agent.decompose_task(concrete_task)
    log_callback("タスクの分解が完了しました")
    output["agents"]["DecomposeTaskAgent"] = {
        "input": concrete_task,
        "output": decomposed_tasks
    }

    output["final_decomposed_tasks"] = decomposed_tasks

    # Output results to JSON file
    log_callback("結果をJSONファイルに保存中...")
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.json"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log_callback(f"結果が {os.path.join(output_dir, filename)} に保存されました。")

if __name__ == "__main__":
    if len(sys.argv) != 18:
        print("Usage: python main.py <task> "
              "<detect_llm_type> <detect_temperature> <detect_max_tokens> "
              "<persona_llm_type> <persona_temperature> <persona_max_tokens> "
              "<interactive_llm_type> <interactive_temperature> <interactive_max_tokens> "
              "<summarize_llm_type> <summarize_temperature> <summarize_max_tokens> "
              "<decompose_llm_type> <decompose_temperature> <decompose_max_tokens> "
              "<output_dir>")
        sys.exit(1)

    main(
        task=sys.argv[1],
        detect_llm_type=sys.argv[2],
        detect_temperature=float(sys.argv[3]),
        detect_max_tokens=int(sys.argv[4]),
        persona_llm_type=sys.argv[5],
        persona_temperature=float(sys.argv[6]),
        persona_max_tokens=int(sys.argv[7]),
        interactive_llm_type=sys.argv[8],
        interactive_temperature=float(sys.argv[9]),
        interactive_max_tokens=int(sys.argv[10]),
        summarize_llm_type=sys.argv[11],
        summarize_temperature=float(sys.argv[12]),
        summarize_max_tokens=int(sys.argv[13]),
        decompose_llm_type=sys.argv[14],
        decompose_temperature=float(sys.argv[15]),
        decompose_max_tokens=int(sys.argv[16]),
        output_dir=sys.argv[17]
    )