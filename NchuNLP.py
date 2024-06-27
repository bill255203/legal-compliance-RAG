import os
from dotenv import load_dotenv
import requests
from requests import post

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API token and Criminal Key from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CRIMINAL_KEY = os.getenv("CRIMINAL_KEY")

# Hugging Face Inference API URL for the specific model
API_URL = "https://api-inference.huggingface.co/models/NchuNLP/Chinese-Question-Answering"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

# Function to query the Hugging Face Inference API
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Function to fetch documents from CopyToaster API
def fetch_documents_from_copytoaster(input_str, count=5):
    url = "https://api.droidtown.co/CopyToaster/API/"
    payload = {
        "username": "legaltech@droidtown.co",  # Do not change this field
        "copytoaster_key": CRIMINAL_KEY,  # Replace with your desired court key
        "category": "臺灣台北地方法院",  # Input the court name you want to search
        "input_str": input_str,  # Input the sentence you want to search
        "count": count  # Optional, default is 15
    }
    response = post(url, json=payload).json()
    
    if response["status"]:
        return response["results"]
    else:
        print("Error fetching documents:", response)
        return []

# Example usage
input_str = "用LINE恐嚇取財"
question = "使用LINE進行恐嚇取財的法律後果是什麼？"

# Fetch documents from the CopyToaster API
documents = fetch_documents_from_copytoaster(input_str, count=5)

# Ensure that each document has the 'document' key
documents_text = "\n".join([doc.get("document", "No content available") for doc in documents])

# Debugging: Print the combined documents text
print("Combined documents text:", documents_text)

# Format the query for the model
formatted_query = f"---------------------\n{documents_text}\n---------------------\n 有鑑於我們提供了以下法律文件的資訊, 請回答: {question}\n"

# Debugging: Print the formatted query
print("Formatted query:", formatted_query)

# Prepare payload for the Hugging Face Inference API
QA_input = {
    "inputs": {
        "question": question,
        "context": documents_text
    }
}

# Generate response using the Hugging Face Inference API
response = query(QA_input)

# Debugging: Print the response from the model
print("Model response:", response)
