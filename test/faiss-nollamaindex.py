import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import requests

# Custom PromptTemplate class
class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, context_str, query_str):
        return self.template.format(context_str=context_str, query_str=query_str)

# Define paths and model
LAW_DIR = 'C:\\law'
VECTOR_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY_MODEL_NAME = 'llama2-chinese'  # You can change this to your preferred model

# Load the tokenizer and model for vectorization
vector_tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL_NAME)
vector_model = AutoModel.from_pretrained(VECTOR_MODEL_NAME)

# Function to read and preprocess JSON files
def read_json_files(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    try:
                        content = ' '.join([item['issueContent']['list'][0]['content'] for item in data['content']])
                        documents.append({
                            'name': data['name'],
                            'date': data['date'],
                            'isAbolished': data['isAbolished'],
                            'history': data['history'],
                            'content': content
                        })
                    except (KeyError, TypeError, IndexError) as e:
                        print(f"Error processing file {file_path}: {e}")
    return documents

# Function to vectorize content using transformer model
def vectorize_documents(documents):
    embeddings = []
    for doc in documents:
        inputs = vector_tokenizer(doc['content'], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = vector_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings), documents

# Read and preprocess JSON files
documents = read_json_files(LAW_DIR)

# Vectorize documents
document_vectors, document_metadata = vectorize_documents(documents)

# Create and populate FAISS index
dimension = document_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_vectors)

# Function to query the vector database and retrieve relevant documents
def query_vector_database(query, top_k=5):
    inputs = vector_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = vector_model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    distances, indices = index.search(np.array([query_vector]), top_k)
    return [document_metadata[i] for i in indices[0]]

# Function to generate prompt using retrieved documents
def generate_prompt_with_context(retrieved_docs, query):
    context = ' '.join([doc['content'] for doc in retrieved_docs])
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    return qa_template.format(context_str=context, query_str=query)

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

# Main function to demonstrate the process
def main():
    query = "What is the regulation about the management of temple properties?"
    retrieved_docs = query_vector_database(query, top_k=5)
    prompt = generate_prompt_with_context(retrieved_docs, query)
    
    # Query the LLM (change model to your desired model if necessary)
    final_response = query_llm(QUERY_MODEL_NAME, prompt)
    print("Final Response:\n", final_response)

if __name__ == "__main__":
    main()
