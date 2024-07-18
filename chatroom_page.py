import streamlit as st
import pyperclip
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
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        st.error("Failed to retrieve models from Groq API")
        return []

# Fetch the models at the start
models = get_groq_models()

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
            
            # Log translation details
            logger.info(f"Translated {len(batch)} sentences from {source_lang} to {target_lang}")
            logger.info(f"Original: {batch}")
            logger.info(f"Translated: {translated_sentences[-len(batch):]}")
        except Exception as e:
            logger.error(f"Error translating batch: {e}")
            translated_sentences.extend(batch)

    return ' '.join(translated_sentences)

def generate_prompt_with_context(retrieved_docs, query, template):
    context = ' '.join([doc['content'] for doc in retrieved_docs])
    # Don't translate the context, use it as is
    qa_template = PromptTemplate(template)
    return qa_template.format(context_str=context, query_str=query)

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

def save_conversation():
    if st.session_state.current_conversation:
        if 'previous_conversations' not in st.session_state:
            st.session_state.previous_conversations = []
        st.session_state.previous_conversations.append(st.session_state.current_conversation)
        st.session_state.current_conversation = []
        st.success("Conversation saved! You can start a new one now.")
        
from memory_agent import MemoryAgent

# Initialize the MemoryAgent (do this outside the chatroom function, perhaps in your main app file)
memory_agent = MemoryAgent()

def chatroom(memory_agent: MemoryAgent):
    st.sidebar.title("Legal Compliance Chatrooms")

    # Sidebar options for different chatrooms
    chatroom = st.sidebar.selectbox("Select Chatroom", ("Legal Document Drafting", "Legal Advice", "Legal Document Review"))

    # Model selection dropdown
    selected_model = st.sidebar.selectbox("Select Model", models)

    st.title(f"{chatroom} Chatroom")

    # Display conversation history
    for exchange in memory_agent.get_current_conversation():
        st.write(f"**You:** {exchange['question']}")
        st.write(f"**Response:** {exchange['response']}")

    # Define different templates for each chatroom
    templates = {
        "Legal Document Drafting": (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Draft a legal document about {query_str} that follows the context above.\n"
        ),
        "Legal Advice": (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Provide legal advice based on the following query: {query_str}\n"
        ),
        "Legal Document Review": (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Review the legal document and provide feedback for the following query: {query_str}\n"
        )
    }
    # User input
    query = st.text_input("Enter your legal compliance question:", key="input_box")

    if query:
        with st.spinner("Processing your query..."):
            try:
                translated_query = translate_content(query, source_lang="zh-TW", target_lang="en")
                logger.info(f"Translated query: {translated_query}")
                retrieved_indices = query_vector_database(translated_query, top_k=2)
                
                # Ensure documents are loaded
                documents = load_documents()
                
                retrieved_docs = [documents[i] for i in retrieved_indices]
                prompt = generate_prompt_with_context(retrieved_docs, translated_query, templates[chatroom])
                
                st.subheader("Generated Prompt:")
                st.text(prompt)
                
                response = query_groq_api(selected_model, prompt)
                
                st.subheader("Groq API Response (English):")
                st.write(response)
                
                # Add a copy button
                if st.button("Copy to Clipboard"):
                    pyperclip.copy(response)
                    st.success("Response copied to clipboard!")
                
                translated_response = translate_content(response, target_lang="zh-TW", source_lang="en")
                
                st.subheader("Translated Response (Traditional Chinese):")
                st.write(translated_response)
                
                # Add a copy button for translated response
                if st.button("Copy Translated Response to Clipboard"):
                    pyperclip.copy(translated_response)
                    st.success("Translated response copied to clipboard!")
                
                # Update conversation history using MemoryAgent
                memory_agent.add_exchange(query, translated_response)
                
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try again or contact support if the problem persists.")

    # Add a button to save the current conversation and start a new one
    if st.button("Save Conversation and Start New"):
        memory_agent.start_new_conversation()
        st.success("New conversation started!")
        st.rerun()

    st.sidebar.title("About")
    st.sidebar.info("This is a RAG system for legal compliance questions using Groq API and Pinecone vector database.")

# Run the chatroom function
if __name__ == "__main__":
    chatroom(memory_agent)