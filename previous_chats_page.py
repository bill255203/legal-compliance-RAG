import streamlit as st

def previous_chats():
    st.title("Previous Chatrooms")
    
    # Initialize previous_conversations if not already present
    if 'previous_conversations' not in st.session_state:
        st.session_state.previous_conversations = []

    # Display previous conversations
    for i, conversation in enumerate(st.session_state.previous_conversations):
        st.write(f"**Chatroom {i + 1}:**")
        for exchange in conversation:
            st.write(f"**You:** {exchange['question']}")
            st.write(f"**Response:** {exchange['response']}")
        st.write("---")
    
    # Button to clear previous chats
    if st.button("Clear Previous Chats"):
        st.session_state.previous_conversations = []
        st.success("Previous chats cleared!")
        st.rerun()

    # Button to go back to the main chatroom
    if st.button("Back to Chatroom"):
        st.query_params["tab"] = "Chatroom"
        st.rerun()