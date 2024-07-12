import streamlit as st
from chatroom_page import chatroom
from previous_chats_page import previous_chats

# Initialize session state for conversations if not already present
if 'previous_conversations' not in st.session_state:
    st.session_state.previous_conversations = []

if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = []

# Read query params to handle tab selection
default_tab = st.query_params.get("tab", "Chatroom")

# Initialize tabs for navigation
tab1, tab2 = st.tabs(["Chatroom", "Previous Chats"])

if default_tab == "Chatroom":
    with tab1:
        chatroom()
    with tab2:
        previous_chats()
else:
    with tab1:
        chatroom()
    with tab2:
        # Store the current conversation before switching to Previous Chats
        if st.session_state.current_conversation:
            st.session_state.previous_conversations.append(st.session_state.current_conversation)
            st.session_state.current_conversation = []
        previous_chats()

# Update query params when switching tabs
if tab1:
    st.query_params["tab"] = "Chatroom"
elif tab2:
    st.query_params["tab"] = "Previous Chats"