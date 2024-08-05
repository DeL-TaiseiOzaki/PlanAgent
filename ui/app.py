import streamlit as st
import sys
import os
import json
from datetime import datetime
from typing import Callable

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

def streamlit_main():
    st.set_page_config(page_title="Agent System UI", layout="wide")
    st.title("エージェントシステム UI")

    # タスク入力
    task = st.text_input("タスクを入力してください:", "スキルアップのためのpython勉強計画を立てたい")

    # エージェント設定
    st.sidebar.header("エージェント設定")
    agent_configs = {}
    for agent in ['detect', 'interactive', 'summarize', 'refine', 'dispatch']:
        st.sidebar.subheader(f"{agent.capitalize()} Agent")
        agent_configs[agent] = {
            'llm_type': st.sidebar.selectbox(f"{agent} LLM Type", ["openai", "anthropic", "groq", "non_api"], key=f"{agent}_llm_type"),
            'temperature': st.sidebar.slider(f"{agent} Temperature", 0.0, 1.0, 0.7, 0.1, key=f"{agent}_temp"),
            'max_tokens': st.sidebar.number_input(f"{agent} Max Tokens", 100, 2000, 1000, 100, key=f"{agent}_tokens")
        }

    # 出力ディレクトリ
    output_dir = st.sidebar.text_input("出力ディレクトリ:", config.DEFAULT_OUTPUT_DIR)

    # Pinecone使用の切り替え
    config.USE_PINECONE = st.sidebar.checkbox("Pineconeを使用する", value=config.USE_PINECONE)

    # 実行ボタン
    if st.button("エージェントシステムを実行"):
        if 'execution_log' not in st.session_state:
            st.session_state.execution_log = ""
        
        log_placeholder = st.empty()

        def update_log(text):
            st.session_state.execution_log += text + "\n"
            log_placeholder.text_area("実行ログ", st.session_state.execution_log, height=300)

        with st.spinner("エージェントシステムを実行中..."):
            run_agent_system(
                task=task,
                detect_llm_type=agent_configs['detect']['llm_type'],
                detect_temperature=agent_configs['detect']['temperature'],
                detect_max_tokens=agent_configs['detect']['max_tokens'],
                interactive_llm_type=agent_configs['interactive']['llm_type'],
                interactive_temperature=agent_configs['interactive']['temperature'],
                interactive_max_tokens=agent_configs['interactive']['max_tokens'],
                summarize_llm_type=agent_configs['summarize']['llm_type'],
                summarize_temperature=agent_configs['summarize']['temperature'],
                summarize_max_tokens=agent_configs['summarize']['max_tokens'],
                refine_llm_type=agent_configs['refine']['llm_type'],
                refine_temperature=agent_configs['refine']['temperature'],
                refine_max_tokens=agent_configs['refine']['max_tokens'],
                dispatch_llm_type=agent_configs['dispatch']['llm_type'],
                dispatch_temperature=agent_configs['dispatch']['temperature'],
                dispatch_max_tokens=agent_configs['dispatch']['max_tokens'],
                output_dir=output_dir,
                log_callback=update_log
            )
        
        st.success("エージェントシステムの実行が完了しました。")
        
        # 結果の表示
        result_files = [f for f in os.listdir(output_dir) if f.startswith("results_") and f.endswith(".json")]
        if result_files:
            latest_result = max(result_files)
            result_file = os.path.join(output_dir, latest_result)
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            st.header("実行結果")
            for agent, data in result['agents'].items():
                with st.expander(f"{agent} の結果"):
                    st.json(data)
            
            st.subheader("最終エージェント指示")
            st.json(result['final_agent_instructions'])
        else:
            st.warning("結果ファイルが見つかりません。")

    # インタラクティブモード
    st.header("インタラクティブモード")
    if 'interactive_agent' not in st.session_state:
        st.session_state.interactive_agent = InteractiveAgent(get_llm(
            agent_configs['interactive']['llm_type'],
            agent_configs['interactive']['temperature'],
            agent_configs['interactive']['max_tokens']
        ))

    # 会話履歴の表示
    for message in st.session_state.interactive_agent.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 質問回数の表示
    st.write(f"質問回数: {st.session_state.interactive_agent.question_count}/{st.session_state.interactive_agent.max_questions}")

    if not st.session_state.interactive_agent.task_understood and st.session_state.interactive_agent.question_count < st.session_state.interactive_agent.max_questions:
        if prompt := st.chat_input("メッセージを入力してください"):
            with st.chat_message("user"):
                st.markdown(prompt)

            # 実際の LLM を使用して応答を生成
            response = st.session_state.interactive_agent.generate(prompt)

            with st.chat_message("assistant"):
                st.markdown(response)

            # 会話履歴の更新（Streamlitの再実行のため）
            st.session_state.conversation_history = st.session_state.interactive_agent.conversation_history
            
            # タスク理解完了または質問回数が上限に達した場合のメッセージ
            if st.session_state.interactive_agent.task_understood:
                st.success("タスクの理解が完了しました。これ以上の質問は必要ありません。")
            elif st.session_state.interactive_agent.question_count >= st.session_state.interactive_agent.max_questions:
                st.warning("最大質問回数に達しました。これまでに収集された情報でタスクを進めます。")
    elif st.session_state.interactive_agent.task_understood:
        st.success("タスクの理解が完了しました。これ以上の質問は必要ありません。")
    else:
        st.warning("最大質問回数に達しました。これまでに収集された情報でタスクを進めます。")

    # リセットボタン
    if st.button("会話をリセット"):
        st.session_state.interactive_agent.reset()
        st.experimental_rerun()

if __name__ == "__main__":
    streamlit_main()