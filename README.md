# PlanAgent

## 概要
PlanAgentは、高度なAI駆動の計画およびスケジューリングツールです。大規模言語モデル（LLM）を活用して、タスク管理を最適化し、自動化します。これにより、複雑なプロジェクトの計画と実行が容易になります。

## 特徴
- **タスクからの計画の自動立案**: 計画を自動で生成．
- **タスクを分解・洗練**: タスクを分解し，必要に応じて洗練
- **最適なエージェントにディスパッチ**: 後続のエージェントへ．

## インストール
1. リポジトリをクローン:
    ```bash
    git clone https://github.com/DeL-TaiseiOzaki/PlanAgent.git
    ```
2. プロジェクトディレクトリに移動:
    ```bash
    cd PlanAgent
    ```
3. 必要なパッケージをインストール:
    ```bash
    pip install -r requirements.txt
    ```
4. .envファイルを用意:
    ```bash
    OPENAI_API_KEY=your-openai-api-key
    ANTHROPIC_API_KEY=your-anthropic-api-key
    GROQ_API_KEY=your-groq-api-key
    ```

## 使い方

config.pyに各エージェントの元になるLLMを指定 \\
その他パラメータも指定

メインスクリプトを実行してプランニングエージェントを起動:
```bash
sh run_main.sh
```

## フォルダー構造

- agents/: コアエージェントロジックを含む。
- llm_interfaces/: 大規模言語モデルとのインターフェース。
- prompts/: エージェントが使用する事前定義されたプロンプト。
- output/: 出力ファイルとログ。