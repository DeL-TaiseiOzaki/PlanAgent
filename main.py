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

def main(task: str, plan_llm_type: str, plan_temperature: float, plan_max_tokens: int,
         refine_llm_type: str, refine_temperature: float, refine_max_tokens: int,
         dispatch_llm_type: str, dispatch_temperature: float, dispatch_max_tokens: int,
         output_dir: str) -> None:
    plan_llm = get_llm(plan_llm_type, plan_temperature, plan_max_tokens)
    refine_llm = get_llm(refine_llm_type, refine_temperature, refine_max_tokens)
    dispatch_llm = get_llm(dispatch_llm_type, dispatch_temperature, dispatch_max_tokens)

    output = {
        "task": task,
        "agents": {}
    }

    # 不足情報の検出
    detect_missinfo_agent = DetectMissinfoAgent(plan_llm)
    missing_info = detect_missinfo_agent.detect_missing_info(task)
    output["agents"]["DetectMissinfoAgent"] = {
        "input": task,
        "output": missing_info
    }

    # インタラクティブな情報収集
    interactive_agent = InteractiveAgent(plan_llm)
    conversation_history = interactive_agent.interactive_information_collection(task, missing_info)
    output["agents"]["InteractiveAgent"] = {
        "input": {
            "task": task,
            "missing_info": missing_info
        },
        "output": conversation_history
    }

    # タスクの要約
    summarize_task_agent = SummarizeTaskAgent(plan_llm)
    task_summary = summarize_task_agent.summarize_task(task, conversation_history)
    output["agents"]["SummarizeTaskAgent"] = {
        "input": {
            "task": task,
            "conversation_history": conversation_history
        },
        "output": task_summary
    }

    #計画を生成・精緻化
    refine_agent = RefineAgent(refine_llm)
    initial_plan = refine_agent.refine_plan(
        task_summary,
        config.MAX_PLAN_REFINE_CHAIN_LENGTH,
        0,
        config.MAX_PLAN_TREE_DEPTH,
        "",  # summaryは不要になったため空文字列を渡す
        []   # 会話履歴を空のリストに変更
    )
    output["agents"]["RefineAgent"] = {
        "input": {
            "task_summary": task_summary,
            "max_plan_refine_chain_length": config.MAX_PLAN_REFINE_CHAIN_LENGTH,
            "max_plan_tree_depth": config.MAX_PLAN_TREE_DEPTH
        },
        "output": initial_plan
    }

    #タスク実行に必要なエージェントを宣言
    # DispatchAgentを使用してタスク実行に必要なエージェントを宣言
    dispatch_agent = DispatchAgent(dispatch_llm)
    dispatch_result = dispatch_agent.dispatch(task_summary, "", str(initial_plan))
    parsed_dispatch_result = parse_dispatch_result(dispatch_result)
    output["agents"]["DispatchAgent"] = {
        "input": {
            "task_summary": task_summary,
            "initial_plan": initial_plan
        },
        "output": dispatch_result
    }

    # 最終的なエージェントリストとその指示を追加
    output["final_agent_instructions"] = parsed_dispatch_result

    # 結果をJSONファイルに出力
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.json"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n結果が {os.path.join(output_dir, filename)} に保存されました。")

if __name__ == "__main__":
    if len(sys.argv) != 12:
        print("Usage: python main.py <task> <plan_llm_type> <plan_temperature> <plan_max_tokens> "
              "<refine_llm_type> <refine_temperature> <refine_max_tokens> "
              "<dispatch_llm_type> <dispatch_temperature> <dispatch_max_tokens> <output_dir>")
        sys.exit(1)

    main(
        task=sys.argv[1],
        plan_llm_type=sys.argv[2],
        plan_temperature=float(sys.argv[3]),
        plan_max_tokens=int(sys.argv[4]),
        refine_llm_type=sys.argv[5],
        refine_temperature=float(sys.argv[6]),
        refine_max_tokens=int(sys.argv[7]),
        dispatch_llm_type=sys.argv[8],
        dispatch_temperature=float(sys.argv[9]),
        dispatch_max_tokens=int(sys.argv[10]),
        output_dir=sys.argv[11]
    )