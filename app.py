import streamlit as st
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import groq

nltk.download('punkt')

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = groq.Client(api_key=GROQ_API_KEY)
credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
translate_client = translate.Client(credentials=credentials)

# Constants
VECTOR_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
QUERY_MODEL_NAME = 'llama3-8b-8192'
INDEX_NAME = 'law-documents'

# Load models
vector_tokenizer = AutoTokenizer.from_pretrained(VECTOR_MODEL_NAME)
vector_model = AutoModel.from_pretrained(VECTOR_MODEL_NAME)
index = pc.Index(INDEX_NAME)

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, context_str, query_str):
        return self.template.format(context_str=context_str, query_str=query_str)

def query_vector_database(query, top_k=2):
    inputs = vector_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = vector_model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
    response = index.query(vector=query_vector, top_k=top_k)
    return [int(match['id']) for match in response['matches']]

def translate_content(content, target_lang="en", source_lang="zh"):
    sentences = sent_tokenize(content)
    translated_sentences = []
    batch_size = 100

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        try:
            results = translate_client.translate(batch, target_language=target_lang, source_language=source_lang)
            translated_sentences.extend([result['translatedText'] for result in results])
        except Exception as e:
            st.error(f"Error translating batch: {e}")
            translated_sentences.extend(batch)

    return ' '.join(translated_sentences)

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

def query_groq_api(model, prompt):
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return response.choices[0].message.content

def load_documents():
    documents = []
    for root, dirs, files in os.walk('C:\\laws'):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        content = extract_content(data)
                        documents.append({
                            'id': file_path.replace("\\", "/"),
                            'name': data.get('name', 'Unknown'),
                            'date': data.get('date', 'Unknown'),
                            'isAbolished': data.get('isAbolished', False),
                            'history': data.get('history', ''),
                            'content': content
                        })
                    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as e:
                        st.error(f"Error processing file {file_path}: {e}")
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

# Streamlit UI
st.title("Legal Compliance RAG System")

# Load documents
documents = load_documents()

# User input
query = st.text_input("Enter your legal compliance question:")

if query:
    with st.spinner("Processing your query..."):
        try:
            # Translate user query to English for processing
            english_query = translate_content(query, source_lang="zh-TW", target_lang="en")
            
            # Retrieve documents using the original Chinese query
            retrieved_indices = query_vector_database(query, top_k=2)
            retrieved_docs = [documents[i] for i in retrieved_indices]
            
            # Generate prompt with translated context and English query
            prompt = generate_prompt_with_context(retrieved_docs, english_query)
            
            st.subheader("Generated Prompt (English):")
            st.text(prompt)
            
            # Query Groq API with English prompt
            response = query_groq_api(QUERY_MODEL_NAME, prompt)
            
            st.subheader("Groq API Response (English):")
            st.write(response)
            
            # Translate response back to Chinese
            translated_response = translate_content(response, target_lang="zh-TW", source_lang="en")
            
            st.subheader("Translated Response (Traditional Chinese):")
            st.write(translated_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or contact support if the problem persists.")

st.sidebar.title("About")
st.sidebar.info("This is a RAG system for legal compliance questions using Groq API and Pinecone vector database.")