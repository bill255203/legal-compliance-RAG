import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import requests
import pickle

# Set the environment variable to bypass the OpenMP runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
INDEX_FILE = 'faiss_index.bin'
METADATA_FILE = 'metadata.pkl'

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
                    try:
                        data = json.load(f)
                        content = extract_content(data)
                        documents.append({
                            'name': data.get('name', 'Unknown'),
                            'date': data.get('date', 'Unknown'),
                            'isAbolished': data.get('isAbolished', False),
                            'history': data.get('history', ''),
                            'content': content
                        })
                    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
                        print(f"Error processing file {file_path}: {e}")
    return documents

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

# Function to vectorize content using transformer model
def vectorize_documents(documents):
    embeddings = []
    for doc in documents:
        inputs = vector_tokenizer(doc['content'], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = vector_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings), documents

# Function to save the FAISS index and metadata
def save_faiss_index(index, metadata, index_file, metadata_file):
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

# Function to load the FAISS index and metadata
def load_faiss_index(index_file, metadata_file):
    index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

# Read and preprocess JSON files
documents = read_json_files(LAW_DIR)

# Vectorize documents
document_vectors, document_metadata = vectorize_documents(documents)

# Create and populate FAISS index
dimension = document_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_vectors)

# Save the FAISS index and metadata
save_faiss_index(index, document_metadata, INDEX_FILE, METADATA_FILE)

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
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("Creating and saving FAISS index...")
        documents = read_json_files(LAW_DIR)
        document_vectors, document_metadata = vectorize_documents(documents)
        dimension = document_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(document_vectors)
        save_faiss_index(index, document_metadata, INDEX_FILE, METADATA_FILE)
    else:
        print("Loading FAISS index from disk...")
        index, document_metadata = load_faiss_index(INDEX_FILE, METADATA_FILE)

    query = "What is the regulation about the management of temple properties?"
    retrieved_docs = query_vector_database(query, top_k=5)
    prompt = generate_prompt_with_context(retrieved_docs, query)
    
    # Query the LLM (change model to your desired model if necessary)
    final_response = query_llm(QUERY_MODEL_NAME, prompt)
    print("Final Response:\n", final_response)

if __name__ == "__main__":
    main()
