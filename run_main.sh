#!/bin/bash

#タスクの設定
TASK="サッカー初心者用の練習メニューを考える"

#LLM_TYPE: "groq" or "openai" or "anthropic" or "non_api"

#PlanAgentの設定
PLAN_LLM_TYPE="anthropic"  
PLAN_TEMPERATURE=0.7
PLAN_MAX_TOKENS=1000

#RefineAgentの設定
REFINE_LLM_TYPE="anthropic"  
REFINE_TEMPERATURE=0.8
REFINE_MAX_TOKENS=1500

#DispatchAgentの設定
DISPATCH_LLM_TYPE="anthropic"  
DISPATCH_TEMPERATURE=0.6
DISPATCH_MAX_TOKENS=800

#出力ディレクトリ
OUTPUT_DIR="./output"

#Pythonスクリプトを実行
python3 main.py "$TASK" \
    "$PLAN_LLM_TYPE" "$PLAN_TEMPERATURE" "$PLAN_MAX_TOKENS" \
    "$REFINE_LLM_TYPE" "$REFINE_TEMPERATURE" "$REFINE_MAX_TOKENS" \
    "$DISPATCH_LLM_TYPE" "$DISPATCH_TEMPERATURE" "$DISPATCH_MAX_TOKENS" \
    "$OUTPUT_DIR"
