import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from requests import post
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
CRIMINAL_KEY = os.getenv("CRIMINAL_KEY")


# Authenticate with Hugging Face
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
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # Combine the context documents into a single context string
    context = "\n\n".join([doc["document"] for doc in context_documents])
    # Create the full prompt
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
    
    # Tokenize the input
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(inputs.input_ids, max_length=512, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Example usage
input_str = "用LINE恐嚇取財"
question = "What are the legal consequences of using LINE for extortion?"

# Fetch documents from the CopyToaster API
documents = fetch_documents_from_copytoaster(input_str, count=5)

# Generate response using the Llama model
response = generate_llama_response(question, documents)

print(response)
