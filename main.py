import sys
import json
import os
import torch
from typing import Dict, Any
from datetime import datetime
from llm_interfaces.base_llm import BaseLLM
from agents.plan_agent import PlanAgent
from agents.dispatch_agent import DispatchAgent
from agents.refine_agent import RefineAgent
import config

def get_llm(llm_type: str, temperature: float, max_tokens: int) -> BaseLLM:
    """
    指定されたタイプのLLMインスタンスを生成して返す。

    Args:
        llm_type (str): LLMのタイプ ("openai", "anthropic", "groq", "non_api")
        temperature (float): 生成時の温度パラメータ
        max_tokens (int): 生成する最大トークン数

    Returns:
        BaseLLM: 指定されたタイプのLLMインスタンス

    Raises:
        ValueError: サポートされていないLLMタイプが指定された場合
    """
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

def main(task: str, plan_llm_type: str, plan_temperature: float, plan_max_tokens: int,
         refine_llm_type: str, refine_temperature: float, refine_max_tokens: int,
         dispatch_llm_type: str, dispatch_temperature: float, dispatch_max_tokens: int,
         output_dir: str) -> None:
    """
    メイン実行関数。タスクの計画、洗練、ディスパッチを行い、結果を出力する。

    Args:
        task (str): 実行するタスクの説明
        plan_llm_type (str): 計画生成に使用するLLMのタイプ
        plan_temperature (float): 計画生成時の温度パラメータ
        plan_max_tokens (int): 計画生成時の最大トークン数
        refine_llm_type (str): 計画洗練に使用するLLMのタイプ
        refine_temperature (float): 計画洗練時の温度パラメータ
        refine_max_tokens (int): 計画洗練時の最大トークン数
        dispatch_llm_type (str): ディスパッチに使用するLLMのタイプ
        dispatch_temperature (float): ディスパッチ時の温度パラメータ
        dispatch_max_tokens (int): ディスパッチ時の最大トークン数
        output_dir (str): 結果を出力するディレクトリパス
    """
    plan_llm = get_llm(plan_llm_type, plan_temperature, plan_max_tokens)
    refine_llm = get_llm(refine_llm_type, refine_temperature, refine_max_tokens)
    dispatch_llm = get_llm(dispatch_llm_type, dispatch_temperature, dispatch_max_tokens)

    plan_agent = PlanAgent(plan_llm)
    refine_agent = RefineAgent(refine_llm)
    dispatch_agent = DispatchAgent(dispatch_llm)

    # 2. PlanAgentが初期計画を生成
    initial_plan = plan_agent.initial_plan_generation(task)

    # 3. RefineAgentがタスクを分解
    subtask_id = "1"
    refine_result = refine_agent.refine_plan(
        subtask_id, 
        config.MAX_PLAN_REFINE_CHAIN_LENGTH, 
        0, 
        config.MAX_PLAN_TREE_DEPTH, 
        "ファイルシステム構造の文字列", 
        "修正ノードメッセージ"
    )

    # 4. DispatchAgentがタスク実行に必要なエージェントを宣言
    dispatch_result = dispatch_agent.dispatch(task, "例示的なシステムプロンプト", "例示的なユーザープロンプト")

     # 結果を出力
    output = {
        "task": task,
        "llm_configs": {
            "plan_agent": {"type": plan_llm_type, "temperature": plan_temperature, "max_tokens": plan_max_tokens},
            "refine_agent": {"type": refine_llm_type, "temperature": refine_temperature, "max_tokens": refine_max_tokens},
            "dispatch_agent": {"type": dispatch_llm_type, "temperature": dispatch_temperature, "max_tokens": dispatch_max_tokens}
        },
        "plan_agent_result": initial_plan,
        "refine_agent_result": refine_result,
        "dispatch_agent_result": dispatch_result
    }

    os.makedirs(output_dir, exist_ok=True)

    #データの出力
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{current_time}.json"
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 12:
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

    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())