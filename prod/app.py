import streamlit as st
from chatroom_page import chatroom
from previous_chats_page import previous_chats
import os
from dotenv import load_dotenv
from memory_agent import MemoryAgent

# Initialize the MemoryAgent
memory_agent = MemoryAgent()

# Load environment variables from .env file
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set Google credentials for the environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Read query params to handle tab selection
default_tab = st.query_params.get("tab", "Chatroom")

# Initialize tabs for navigation
tab1, tab2 = st.tabs(["Chatroom", "Previous Chats"])

if default_tab == "Chatroom":
    with tab1:
        chatroom(memory_agent)
    with tab2:
        previous_chats(memory_agent)
else:
    with tab1:
        chatroom(memory_agent)
    with tab2:
        previous_chats(memory_agent)

# Update query params when switching tabs
if tab1:
    st.query_params["tab"] = "Chatroom"
elif tab2:
    st.query_params["tab"] = "Previous Chats"