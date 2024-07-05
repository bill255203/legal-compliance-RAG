import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import spacy

# Load environment variables from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define paths and model
LAW_DIR = 'C:\\law'
VECTOR_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Load the tokenizer and model for vectorization
vector_tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL_NAME)
vector_model = AutoModel.from_pretrained(VECTOR_MODEL_NAME)

# Create Pinecone index
index_name = 'law-documents'
dimension = 384  # Replace with the dimension of your embeddings
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

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
                            'id': file_path,
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

# Function to translate content using GoogleTranslator
def translate_content(content, target_language='en'):
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(content)

# Function to perform NER using spaCy
def perform_ner(documents):
    for doc in documents:
        spacy_doc = nlp(doc['content'])
        entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
        doc['entities'] = entities
    return documents

# Function to vectorize content using transformer model
def vectorize_documents(documents):
    embeddings = []
    for doc in documents:
        translated_content = translate_content(doc['content'])
        inputs = vector_tokenizer(translated_content, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = vector_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        doc['content'] = translated_content  # Optionally store the translated content back in the document
    return np.array(embeddings), documents

# Read and preprocess JSON files
documents = read_json_files(LAW_DIR)

# Perform Named Entity Recognition (NER)
documents = perform_ner(documents)

# Vectorize documents
document_vectors, document_metadata = vectorize_documents(documents)

# Upload vectors to Pinecone
vectors = [(str(i), vector) for i, vector in enumerate(document_vectors)]
index.upsert(vectors)

print("Indexing complete.")
