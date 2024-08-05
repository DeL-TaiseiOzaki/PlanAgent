import sys
import json
import os
from datetime import datetime
from typing import List, Dict
from llm_interfaces.base_llm import BaseLLM
from agents.detect_missinfo_agent import DetectMissinfoAgent
from agents.interactive_agent import InteractiveAgent
from agents.summarize_task_agent import SummarizeTaskAgent
from agents.refine_agent import RefineAgent
from agents.dispatch_agent import DispatchAgent
import config

# Pinecone関連のインポートは条件付きで行う
if config.USE_PINECONE:
    from ui.pinecone_utils import init_vector_store, upsert_to_store, query_store

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

def parse_dispatch_result(dispatch_result: str) -> List[Dict[str, str]]:
    agents = []
    lines = dispatch_result.strip().split("\n")
    current_agent = {}
    
    for line in lines:
        if line.startswith("エージェント"):
            if current_agent:
                agents.append(current_agent)
            current_agent = {"name": line.split(":")[1].strip()}
        elif line.startswith("指示:"):
            current_agent["instruction"] = line.split(":", 1)[1].strip()
    
    if current_agent:
        agents.append(current_agent)
    
    return agents

def main(task: str, 
         detect_llm_type: str, detect_temperature: float, detect_max_tokens: int,
         interactive_llm_type: str, interactive_temperature: float, interactive_max_tokens: int,
         summarize_llm_type: str, summarize_temperature: float, summarize_max_tokens: int,
         refine_llm_type: str, refine_temperature: float, refine_max_tokens: int,
         dispatch_llm_type: str, dispatch_temperature: float, dispatch_max_tokens: int,
         output_dir: str) -> None:

    # ベクトルストアの初期化（必要な場合のみ）
    if config.USE_PINECONE:
        vector_store = init_vector_store()

    detect_llm = get_llm(detect_llm_type, detect_temperature, detect_max_tokens)
    interactive_llm = get_llm(interactive_llm_type, interactive_temperature, interactive_max_tokens)
    summarize_llm = get_llm(summarize_llm_type, summarize_temperature, summarize_max_tokens)
    refine_llm = get_llm(refine_llm_type, refine_temperature, refine_max_tokens)
    dispatch_llm = get_llm(dispatch_llm_type, dispatch_temperature, dispatch_max_tokens)

    output = {
        "task": task,
        "agents": {}
    }

    # DetectMissinfoAgent
    detect_missinfo_agent = DetectMissinfoAgent(detect_llm)
    detect_prompt = detect_missinfo_agent.system_prompt + "\n\n" + detect_missinfo_agent.user_prompt.replace("{{task}}", task)
    missing_info = detect_missinfo_agent.detect_missing_info(task)
    output["agents"]["DetectMissinfoAgent"] = {
        "prompt": detect_prompt,
        "output": missing_info
    }

    # ベクトルストアにタスクと不足情報を保存（Pinecone使用時のみ）
    if config.USE_PINECONE:
        task_vector = detect_llm.get_embeddings(task)
        upsert_to_store(vector_store, [(task, task_vector, {"missing_info": missing_info})])

    # InteractiveAgent
    interactive_agent = InteractiveAgent(interactive_llm)
    interactive_prompt = interactive_agent.system_prompt + "\n\n" + interactive_agent.user_prompt.replace("{{task}}", task).replace("{{missing_info}}", str(missing_info))
    conversation_history = interactive_agent.interactive_information_collection(task, missing_info)
    output["agents"]["InteractiveAgent"] = {
        "prompt": interactive_prompt,
        "output": conversation_history
    }

    # SummarizeTaskAgent
    summarize_task_agent = SummarizeTaskAgent(summarize_llm)
    summarize_prompt = summarize_task_agent.system_prompt + "\n\n" + summarize_task_agent.user_prompt.replace("{{task}}", task).replace("{{conversation_history}}", str(conversation_history))
    task_summary = summarize_task_agent.summarize_task(task, conversation_history)
    output["agents"]["SummarizeTaskAgent"] = {
        "prompt": summarize_prompt,
        "output": task_summary
    }

    # RefineAgent
    refine_agent = RefineAgent(refine_llm)
    refine_prompt = refine_agent.system_prompt + "\n\n" + refine_agent.user_prompt.replace("{{task}}", task_summary).replace("{{max_step}}", str(config.MAX_PLAN_REFINE_CHAIN_LENGTH)).replace("{{modify_steps}}", "0").replace("{{max_plan_tree_depth}}", str(config.MAX_PLAN_TREE_DEPTH)).replace("{{summary}}", "").replace("{{conversation_history}}", "[]")
    initial_plan = refine_agent.refine_plan(
        task_summary,
        config.MAX_PLAN_REFINE_CHAIN_LENGTH,
        0,
        config.MAX_PLAN_TREE_DEPTH,
        "",
        []
    )
    
    # ベクトルストアから関連情報を取得（Pinecone使用時のみ）
    if config.USE_PINECONE:
        related_info = query_store(vector_store, refine_llm.get_embeddings(task_summary))
    else:
        related_info = []

    # 関連情報を使用して計画を改善
    improved_plan = refine_agent.refine_plan(
        task_summary,
        config.MAX_PLAN_REFINE_CHAIN_LENGTH,
        0,
        config.MAX_PLAN_TREE_DEPTH,
        "",
        related_info
    )
    output["agents"]["RefineAgent"] = {
        "prompt": refine_prompt,
        "output": improved_plan
    }

    # DispatchAgent
    dispatch_agent = DispatchAgent(dispatch_llm)
    dispatch_prompt = dispatch_agent.system_prompt + "\n\n" + dispatch_agent.user_prompt.replace("{{task}}", task_summary).replace("{{summary}}", "").replace("{{initial_plan}}", str(improved_plan))
    dispatch_result = dispatch_agent.dispatch(task_summary, "", str(improved_plan))
    parsed_dispatch_result = parse_dispatch_result(dispatch_result)
    output["agents"]["DispatchAgent"] = {
        "prompt": dispatch_prompt,
        "output": dispatch_result
    }

    output["final_agent_instructions"] = parsed_dispatch_result

    # Output results to JSON file
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.json"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果が {os.path.join(output_dir, filename)} に保存されました。")

if __name__ == "__main__":
    if len(sys.argv) != 18:
        print("Usage: python main.py <task> "
              "<detect_llm_type> <detect_temperature> <detect_max_tokens> "
              "<interactive_llm_type> <interactive_temperature> <interactive_max_tokens> "
              "<summarize_llm_type> <summarize_temperature> <summarize_max_tokens> "
              "<refine_llm_type> <refine_temperature> <refine_max_tokens> "
              "<dispatch_llm_type> <dispatch_temperature> <dispatch_max_tokens> "
              "<output_dir>")
        sys.exit(1)

    main(
        task=sys.argv[1],
        detect_llm_type=sys.argv[2],
        detect_temperature=float(sys.argv[3]),
        detect_max_tokens=int(sys.argv[4]),
        interactive_llm_type=sys.argv[5],
        interactive_temperature=float(sys.argv[6]),
        interactive_max_tokens=int(sys.argv[7]),
        summarize_llm_type=sys.argv[8],
        summarize_temperature=float(sys.argv[9]),
        summarize_max_tokens=int(sys.argv[10]),
        refine_llm_type=sys.argv[11],
        refine_temperature=float(sys.argv[12]),
        refine_max_tokens=int(sys.argv[13]),
        dispatch_llm_type=sys.argv[14],
        dispatch_temperature=float(sys.argv[15]),
        dispatch_max_tokens=int(sys.argv[16]),
        output_dir=sys.argv[17]
    )