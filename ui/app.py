import streamlit as st
import sys
import os
import json
from datetime import datetime
from typing import Callable
import asyncio

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main as run_agent_system, get_llm
from agents.interactive_agent import InteractiveAgent
import config

def streamlit_print(*args, **kwargs):
    output = " ".join(map(str, args))
    st.text(output)

import builtins
builtins.print = streamlit_print

async def run_agent_system_async(task, detect_llm_type, detect_temperature, detect_max_tokens,
                                 persona_llm_type, persona_temperature, persona_max_tokens,
                                 interactive_llm_type, interactive_temperature, interactive_max_tokens,
                                 summarize_llm_type, summarize_temperature, summarize_max_tokens,
                                 decompose_llm_type, decompose_temperature, decompose_max_tokens,
                                 output_dir, log_callback):
    return await asyncio.to_thread(run_agent_system, task, detect_llm_type, detect_temperature, detect_max_tokens,
                                   persona_llm_type, persona_temperature, persona_max_tokens,
                                   interactive_llm_type, interactive_temperature, interactive_max_tokens,
                                   summarize_llm_type, summarize_temperature, summarize_max_tokens,
                                   decompose_llm_type, decompose_temperature, decompose_max_tokens,
                                   output_dir, log_callback)

def streamlit_main():
    st.set_page_config(page_title="Agent System UI", layout="wide")
    st.title("エージェントシステム UI")

    # セッション状態の初期化
    if 'state' not in st.session_state:
        st.session_state.state = 'input'
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = ""
    if 'result' not in st.session_state:
        st.session_state.result = None

    # タスク入力
    task = st.text_input("タスクを入力してください:", "スキルアップのためのpython勉強計画を立てたい")

    # エージェント設定
    st.sidebar.header("エージェント設定")
    agent_configs = {}
    for agent in ['detect', 'persona', 'interactive', 'summarize', 'decompose']:
        st.sidebar.subheader(f"{agent.capitalize()} Agent")
        agent_configs[agent] = {
            'llm_type': st.sidebar.selectbox(f"{agent} LLM Type", ["anthropic"], key=f"{agent}_llm_type"),
            'temperature': st.sidebar.slider(f"{agent} Temperature", 0.0, 1.0, 0.7, 0.1, key=f"{agent}_temp"),
            'max_tokens': st.sidebar.number_input(f"{agent} Max Tokens", 100, 2000, 1000, 100, key=f"{agent}_tokens")
        }

    # 出力ディレクトリ
    output_dir = st.sidebar.text_input("出力ディレクトリ:", config.DEFAULT_OUTPUT_DIR)

    # ユーザーID設定
    config.CURRENT_USER_ID = st.sidebar.number_input("現在のユーザーID", min_value=1, max_value=5, value=config.CURRENT_USER_ID)

    # 実行ログの表示
    log_placeholder = st.empty()

    def update_log(text):
        st.session_state.execution_log += text + "\n"
        log_placeholder.text_area("実行ログ", st.session_state.execution_log, height=300)

    # 実行ボタン
    if st.button("エージェントシステムを実行") and st.session_state.state == 'input':
        st.session_state.state = 'running'
        st.session_state.execution_log = ""
        st.rerun()

    # エージェントシステムの実行
    if st.session_state.state == 'running':
        with st.spinner("エージェントシステムを実行中..."):
            result = asyncio.run(run_agent_system_async(
                task, 
                agent_configs['detect']['llm_type'],
                agent_configs['detect']['temperature'],
                agent_configs['detect']['max_tokens'],
                agent_configs['persona']['llm_type'],
                agent_configs['persona']['temperature'],
                agent_configs['persona']['max_tokens'],
                agent_configs['interactive']['llm_type'],
                agent_configs['interactive']['temperature'],
                agent_configs['interactive']['max_tokens'],
                agent_configs['summarize']['llm_type'],
                agent_configs['summarize']['temperature'],
                agent_configs['summarize']['max_tokens'],
                agent_configs['decompose']['llm_type'],
                agent_configs['decompose']['temperature'],
                agent_configs['decompose']['max_tokens'],
                output_dir,
                update_log
            ))
        st.session_state.state = 'completed'
        st.session_state.result = result
        st.rerun()

    # 結果の表示
    if st.session_state.state == 'completed':
        st.success("エージェントシステムの実行が完了しました。")
        
        if st.session_state.result:
            st.header("実行結果")
            for agent, data in st.session_state.result['agents'].items():
                with st.expander(f"{agent} の結果"):
                    st.json(data)
            
            st.subheader("最終分解タスク")
            st.json(st.session_state.result['final_decomposed_tasks'])
        else:
            st.warning("結果が見つかりません。")

    # リセットボタン
    if st.button("システムをリセット"):
        st.session_state.state = 'input'
        st.session_state.execution_log = ""
        st.session_state.result = None
        st.rerun()

if __name__ == "__main__":
    streamlit_main()