import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
import requests
from dotenv import load_dotenv
from translate import Translator
import time
import re

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Custom PromptTemplate class
class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, context_str, query_str):
        return self.template.format(context_str=context_str, query_str=query_str)

# Define paths and model
VECTOR_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY_MODEL_NAME = 'llama2-chinese'  # You can change this to your preferred model

# Load the tokenizer and model for vectorization
vector_tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL_NAME)
vector_model = AutoModel.from_pretrained(VECTOR_MODEL_NAME)

# Connect to the Pinecone index
index_name = 'law-documents'
index = pc.Index(index_name)

# Function to query the vector database and retrieve relevant documents
def query_vector_database(query, top_k=5):
    inputs = vector_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = vector_model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()  # Convert to list
    response = index.query(vector=query_vector, top_k=top_k)
    return [int(match['id']) for match in response['matches']]  # Convert to integer indices

# Function to chunk large text
def smart_chunk_text(text, chunk_size=512, overlap=50):
    # Split the text into sentences
    if re.search('[\u4e00-\u9fff]', text):  # Check if the text is Chinese
        sentences = re.split(r'(?<=。)', text)
    else:  # Assume English text
        sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence if current_chunk == "" else ' ' + sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Function to translate content using Translator and chunking if necessary
def translate_content(content, to_lang="en", from_lang="zh-TW"):
    translator = Translator(to_lang=to_lang, from_lang=from_lang)
    chunks = smart_chunk_text(content)
    translated_chunks = []
    for chunk in chunks:
        print(f"Translating chunk: {chunk[:30]}...")  # Debugging: show start of the chunk
        try:
            translated_chunk = translator.translate(chunk)
            translated_chunks.append(translated_chunk)
        except Exception as e:
            print(f"Error translating chunk: {e}")
            translated_chunks.append(chunk)  # Fallback to original chunk if translation fails
    return ' '.join(translated_chunks)

# Function to generate prompt using retrieved documents
def generate_prompt_with_context(retrieved_docs, query):
    context = ' '.join([doc['content'] for doc in retrieved_docs])
    translated_context = translate_content(context)
    
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Generate a legal compliance document about {query_str} that has similar format as the context above\n"
    )
    qa_template = PromptTemplate(template)
    return qa_template.format(context_str=translated_context, query_str=query)

# Function to send the prompt to the LLM and get a response
def query_llm(model, prompt):
    url = f"http://localhost:11434/api/generate"  # Change URL if needed
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return json.loads(response.text)['response']
    else:
        print("Error:", response.status_code, response.text)
        return None

# Example translation from English to Chinese
def translate_content_english_to_chinese(content):
    return translate_content(content, to_lang="zh-TW", from_lang="en")

# Example translation from Chinese to English
def translate_content_chinese_to_english(content):
    return translate_content(content, to_lang="en", from_lang="zh-TW")

# Main function to demonstrate the process
def main():
    query = "What is the regulation about the management of temple properties?"
    retrieved_indices = query_vector_database(query, top_k=5)

    # Load documents to get the content for the retrieved IDs
    documents = []
    for root, dirs, files in os.walk('C:\\law'):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        content = extract_content(data)
                        documents.append({
                            'id': file_path.replace("\\", "/"),  # Ensure consistent format
                            'name': data.get('name', 'Unknown'),
                            'date': data.get('date', 'Unknown'),
                            'isAbolished': data.get('isAbolished', False),
                            'history': data.get('history', ''),
                            'content': content
                        })
                    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
                        print(f"Error processing file {file_path}: {e}")

    retrieved_docs = [documents[i] for i in retrieved_indices]  # Map indices to document entries
    prompt = generate_prompt_with_context(retrieved_docs, query)
    print("Prompt:\n", prompt)
    
    # Query the LLM (change model to your desired model if necessary)
    final_response = query_llm(QUERY_MODEL_NAME, prompt)
    print("Final Response:\n", final_response)
    if final_response:
        translated_final_response = translate_content_english_to_chinese(final_response)
        print("Translated Final Response:\n", translated_final_response)

def extract_content(data):
    content = []
    try:
        for item in data.get('content', []):
            issue_content = item.get('issueContent', {})
            if 'list' in issue_content:
                for subitem in issue_content['list']:
                    if 'content' in subitem:
                        content.append(subitem['content'])
                    if 'child' in subitem:
                        for child_item in subitem['child'].get('list', []):
                            if 'content' in child_item:
                                content.append(child_item['content'])
    except (KeyError, TypeError):
        pass
    return ' '.join(content)

if __name__ == "__main__":
    main()
