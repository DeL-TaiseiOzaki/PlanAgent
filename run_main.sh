#!/bin/bash

# タスクの設定
TASK="今日なにか面白いニュースない？"

# LLM_TYPE: "openai" or "anthropic" or "groq" or "non_api"

# DetectMissinfoAgentの設定
DETECT_LLM_TYPE="openai"
DETECT_TEMPERATURE=0.7
DETECT_MAX_TOKENS=1000

# PersonaBasedInfoRetrievalAgentの設定
PERSONA_LLM_TYPE="openai"
PERSONA_TEMPERATURE=0.7
PERSONA_MAX_TOKENS=1000

# InteractiveAgentの設定
INTERACTIVE_LLM_TYPE="openai"
INTERACTIVE_TEMPERATURE=0.7
INTERACTIVE_MAX_TOKENS=1000

# SummarizeTaskAgentの設定
SUMMARIZE_LLM_TYPE="openai"
SUMMARIZE_TEMPERATURE=0.7
SUMMARIZE_MAX_TOKENS=1000

# DecomposeTaskAgentの設定
DECOMPOSE_LLM_TYPE="openai"
DECOMPOSE_TEMPERATURE=0.7
DECOMPOSE_MAX_TOKENS=1000

# 出力ディレクトリ
OUTPUT_DIR="./output"

# Pythonスクリプトを実行
python3 main.py "$TASK" \
    "$DETECT_LLM_TYPE" "$DETECT_TEMPERATURE" "$DETECT_MAX_TOKENS" \
    "$PERSONA_LLM_TYPE" "$PERSONA_TEMPERATURE" "$PERSONA_MAX_TOKENS" \
    "$INTERACTIVE_LLM_TYPE" "$INTERACTIVE_TEMPERATURE" "$INTERACTIVE_MAX_TOKENS" \
    "$SUMMARIZE_LLM_TYPE" "$SUMMARIZE_TEMPERATURE" "$SUMMARIZE_MAX_TOKENS" \
    "$DECOMPOSE_LLM_TYPE" "$DECOMPOSE_TEMPERATURE" "$DECOMPOSE_MAX_TOKENS" \
    "$OUTPUT_DIR"