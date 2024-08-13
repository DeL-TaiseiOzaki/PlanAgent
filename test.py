import requests
from openai import OpenAI

# Google Custom Search APIの設定
google_api_key = 'AIzaSyAwzFUzrsHok2XQ9M-xi_IrBLDSMEIoU_g'
cx = '55b6220724ab7448e'

# OpenAI APIの設定
client = OpenAI(api_key = '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')

def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={cx}"
    response = requests.get(url)
    search_results = response.json()
    return search_results['items'][:3]  # 上位3つの結果を返す

def ask_llm(question, context):
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        max_tokens=150,
        stream=False,
    )
    return response.choices[0].message.content

def search_and_answer(question):
    search_results = google_search(question)
    print(search_results)
    context = "\n".join([result['snippet'] for result in search_results])
    answer = ask_llm(question, context)
    return answer

# 例として質問を検索して回答
question = "2023年　人工知能学会　優秀賞　大阪公立大学　学生"
answer = search_and_answer(question)
print(answer)

