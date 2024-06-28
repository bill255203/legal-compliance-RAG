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
from groq import Groq

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
client = Groq(api_key=GROQ_API_KEY)

# Custom PromptTemplate class
class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, context_str, query_str):
        return self.template.format(context_str=context_str, query_str=query_str)

# Define paths and model
VECTOR_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY_MODEL_NAME = 'mixtral-8x7b-32768'  # Using Groq model

# Load the tokenizer and model for vectorization
vector_tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL_NAME)
vector_model = AutoModel.from_pretrained(VECTOR_MODEL_NAME)

# Connect to the Pinecone index
index_name = 'law-documents'
index = pc.Index(index_name)

# Function to query the vector database and retrieve relevant documents
def query_vector_database(query, top_k=1):
    inputs = vector_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = vector_model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()  # Convert to list
    response = index.query(vector=query_vector, top_k=top_k)
    return [int(match['id']) for match in response['matches']]  # Convert to integer indices

# Function to chunk large text
def chunk_text(text, chunk_size=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to translate content using Translator and chunking if necessary
def translate_content(content, to_lang="en", from_lang="zh-TW"):
    translator = Translator(to_lang=to_lang, from_lang=from_lang)
    chunks = chunk_text(content)
    translated_chunks = []
    for chunk in chunks:
        print(chunk)
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

# Function to send the prompt to Groq API and get a response
def query_groq_api(model, prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return response.choices[0].message.content

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
    
    # Query the Groq API
    final_response = query_groq_api(QUERY_MODEL_NAME, prompt)
    print("Final Response:\n", final_response)
    
    if final_response:
        translated_final_response = translate_content(final_response, to_lang="zh-TW", from_lang="en")
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
