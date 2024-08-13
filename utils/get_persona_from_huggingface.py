from datasets import load_dataset

# データセットをロードする
persona_dataset = load_dataset("MKJ-TOE/persona_dataset5")
conversation_history_dataset = load_dataset("MKJ-TOE/conversation_history_dataset")

def extract_user_persona(user_id):
    # IDが一致するユーザーのペルソナ情報を抽出
    persona_data = None
    for entry in persona_dataset['train']:
        if entry['ID'] == user_id:
            persona_data = {
                'Full_Name': entry['Full_Name'],
                'Age': entry['Age'],
                'Work': entry['Work'],
                'Character': entry['Character'],
                'SoV': entry['SoV']
            }
            break
    return persona_data

def extract_user_conversation_history(user_id):
    # 指定されたユーザーIDの会話履歴を抽出する
    user_conversation_history = []
    for entry in conversation_history_dataset['train']:
        if entry['ID'] == user_id:
            user_conversation_history.append(entry['history'])
    
    return user_conversation_history

def generate_system_prompt(user_id):
    # ユーザーデータを抽出
    persona = extract_user_persona(user_id)
    conversation_history = extract_user_conversation_history(user_id)
    
    if not persona:
        return "User persona not found."
    
    # システムプロンプトを作成
    system_prompt = (
        f"User Persona:\n"
        f"- Full Name: {persona['Full_Name']}\n"
        f"- Age: {persona['Age']}\n"
        f"- Work: {persona['Work']}\n"
        f"- Character: {persona['Character']}\n"
        f"- SoV: {persona['SoV']}\n"
        f"Conversation History:\n"
    )
    
    for conversation in conversation_history:
        system_prompt += f"- {conversation}\n"
    
    return system_prompt

# 例として特定のユーザーIDを指定してプロンプトを生成
user_id = 1  # ここに対象のユーザーIDを指定
system_prompt = generate_system_prompt(user_id)

# 結果を表示
print(system_prompt)