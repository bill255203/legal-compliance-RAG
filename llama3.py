import os
from dotenv import load_dotenv
from requests import post
import ollama  # Import Ollama's SDK (replace with the actual import if different)

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CRIMINAL_KEY = os.getenv("CRIMINAL_KEY")

# Authenticate with Hugging Face (if still needed)
from huggingface_hub import login
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)

def fetch_documents_from_copytoaster(input_str, count=5):
    url = "https://api.droidtown.co/CopyToaster/API/"
    payload = {
        "username": "legaltech@droidtown.co", # Do not change this field
        "copytoaster_key": CRIMINAL_KEY, # Replace with your desired court key
        "category": "臺灣台北地方法院", # Input the court name you want to search
        "input_str": input_str, # Input the sentence you want to search
        "count": count # Optional, default is 15
    }
    response = post(url, json=payload).json()
    if response["status"]:
        return response["results"]
    else:
        return []

def generate_llama_response(prompt, context_documents):
    # Combine the context documents into a single context string
    context = "\n\n".join([doc["document"] for doc in context_documents])
    # Create the full prompt
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
    
    # Generate response using Ollama's Llama3 model
    client = ollama.Client(api_key=HUGGINGFACE_TOKEN)  # Replace with actual initialization if different
    response = client.generate(
        model="llama3",
        prompt=full_prompt,
        max_tokens=512
    )
    
    return response["choices"][0]["text"].strip()

# Example usage
input_str = "用LINE恐嚇取財"
question = "What are the legal consequences of using LINE for extortion?"

# Fetch documents from the CopyToaster API
documents = fetch_documents_from_copytoaster(input_str, count=5)

# Generate response using the Llama model
response = generate_llama_response(question, documents)

print(response)
