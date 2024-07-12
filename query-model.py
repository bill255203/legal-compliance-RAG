import requests
import os

def get_groq_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        models = [model['id'] for model in data['data']]
        return models
    else:
        return []
print(get_groq_models())