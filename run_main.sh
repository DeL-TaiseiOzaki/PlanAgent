#!/bin/bash

# タスクの設定
TASK="スキルアップのためのpython勉強計画を立てたい"

# LLM_TYPE: "groq" or "openai" or "anthropic" or "non_api"

# DetectMissinfoAgentの設定
DETECT_LLM_TYPE="anthropic"
DETECT_TEMPERATURE=0.7
DETECT_MAX_TOKENS=1000

# InteractiveAgentの設定
INTERACTIVE_LLM_TYPE="anthropic"
INTERACTIVE_TEMPERATURE=0.7
INTERACTIVE_MAX_TOKENS=1000

# SummarizeTaskAgentの設定
SUMMARIZE_LLM_TYPE="anthropic"
SUMMARIZE_TEMPERATURE=0.7
SUMMARIZE_MAX_TOKENS=1000

# RefineAgentの設定
REFINE_LLM_TYPE="anthropic"
REFINE_TEMPERATURE=0.8
REFINE_MAX_TOKENS=1500

# DispatchAgentの設定
DISPATCH_LLM_TYPE="anthropic"
DISPATCH_TEMPERATURE=0.6
DISPATCH_MAX_TOKENS=800

# 出力ディレクトリ
OUTPUT_DIR="./output"

# Pythonスクリプトを実行
python3 main.py "$TASK" \
    "$DETECT_LLM_TYPE" "$DETECT_TEMPERATURE" "$DETECT_MAX_TOKENS" \
    "$INTERACTIVE_LLM_TYPE" "$INTERACTIVE_TEMPERATURE" "$INTERACTIVE_MAX_TOKENS" \
    "$SUMMARIZE_LLM_TYPE" "$SUMMARIZE_TEMPERATURE" "$SUMMARIZE_MAX_TOKENS" \
    "$REFINE_LLM_TYPE" "$REFINE_TEMPERATURE" "$REFINE_MAX_TOKENS" \
    "$DISPATCH_LLM_TYPE" "$DISPATCH_TEMPERATURE" "$DISPATCH_MAX_TOKENS" \
    "$OUTPUT_DIR"