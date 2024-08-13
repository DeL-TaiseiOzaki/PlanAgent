from ..base_agent import BaseAgent
import requests
import config

class WebSearchAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, "ToolAgents/web_search_agent_prompt.json")
        self.api_key = config.GOOGLE_SEARCH_API_KEY
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query, num_results=3):
        params = {
            "key": self.api_key,
            "cx": config.GOOGLE_SEARCH_ENGINE_ID, 
            "q": query,
            "num": num_results
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        search_results = response.json()

        # エラーチェック
        if 'error' in search_results:
            raise ValueError(f"Google Search API error: {search_results['error']['message']}")

        # LLMを使用して検索結果を要約
        summary_prompt = self.format_summary_prompt(query, search_results)
        summary = self.generate([
            {"role": "system", "content": "You are a helpful AI assistant that summarizes web search results."},
            {"role": "user", "content": summary_prompt}
        ])

        return {
            "query": query,
            "raw_results": search_results,
            "summary": summary
        }

    def format_summary_prompt(self, query, search_results):
        formatted_results = "\n".join([f"Title: {item.get('title', 'N/A')}\nSnippet: {item.get('snippet', 'N/A')}" 
                                       for item in search_results.get('items', [])])
        return f"Summarize the following search results for the query '{query}':\n\n{formatted_results}"

    def generate(self, messages):
        return self.llm.generate(messages)