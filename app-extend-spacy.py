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
import spacy

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

# Load spaCy model
nlp = spacy.load("zh_core_web_sm")  # Change to appropriate Chinese model

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
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return response['matches']

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

def generate_prompt_with_context(retrieved_docs, query, template):
    context = ' '.join([doc['metadata']['content'] for doc in retrieved_docs])
    translated_context = translate_content(context)
    
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

def perform_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def segment_text(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Streamlit UI
st.sidebar.title("Legal Compliance Chatrooms")

# Sidebar options for different chatrooms
chatroom = st.sidebar.radio("Select Chatroom", ("Legal Document Drafting", "Legal Advice", "Legal Document Review"))

st.title(f"{chatroom} Chatroom")

# Initialize conversation state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

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

# Display conversation history
for exchange in st.session_state.conversation:
    st.write(f"**You:** {exchange['question']}")
    st.write(f"**Response:** {exchange['response']}")

# User input
query = st.text_input("Enter your legal compliance question:", key="input_box")

if query:
    with st.spinner("Processing your query..."):
        try:
            # Perform NER on the query
            query_entities = perform_ner(query)
            st.subheader("Named Entities in Query:")
            st.write(query_entities)

            translated_query = translate_content(query, source_lang="zh-TW", target_lang="en")
            retrieved_docs = query_vector_database(translated_query, top_k=2)
            
            st.subheader("Retrieved Documents:")
            for doc in retrieved_docs:
                st.write(f"Document ID: {doc['id']}")
                st.write(f"Score: {doc['score']}")
                st.write(f"Content: {doc['metadata']['content'][:200]}...")  # Display first 200 characters
                st.write(f"Entities: {doc['metadata'].get('entities', 'Not available')}")
                st.write(f"Sentences: {doc['metadata'].get('sentences', 'Not available')[:3]}")  # Display first 3 sentences
                st.write("---")

            prompt = generate_prompt_with_context(retrieved_docs, translated_query, templates[chatroom])
            
            st.subheader("Generated Prompt:")
            st.text(prompt)
            
            response = query_groq_api(QUERY_MODEL_NAME, prompt)
            
            st.subheader("Groq API Response (English):")
            st.write(response)
            
            translated_response = translate_content(response, target_lang="zh-TW", source_lang="en")
            
            st.subheader("Translated Response (Traditional Chinese):")
            st.write(translated_response)
            
            # Update conversation history
            st.session_state.conversation.append({
                "question": query,
                "response": translated_response
            })
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or contact support if the problem persists.")

st.sidebar.title("About")
st.sidebar.info("This is a RAG system for legal compliance questions using Groq API and Pinecone vector database.")