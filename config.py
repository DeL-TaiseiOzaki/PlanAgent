import os
from dotenv import load_dotenv

load_dotenv()

#APIKey取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#モデル設定
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620" 
GROQ_MODEL = "llama3-70b-8192"  
NON_API_MODEL_PATH = "hbx/Mistral-Interact"
USE_VLLM = True 

#LLM選択
DEFAULT_LLM_TYPE = "openai"  #デフォルトのLLMタイプ

#一般設定
MAX_PLAN_TREE_WIDTH = 4  #計画ツリーの最大幅
MAX_PLAN_TREE_DEPTH = 3  #計画ツリーの最大深さ
MAX_PLAN_REFINE_CHAIN_LENGTH = 5  #計画洗練チェーンの最大長

#出力設定
ENABLE_SUMMARY = True  #要約機能を有効にするかどうか

#デフォルトの温度とトークン設定
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

#出力ディレクトリ
DEFAULT_OUTPUT_DIR = "./output"