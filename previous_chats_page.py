import streamlit as st
from memory_agent import MemoryAgent

import streamlit as st
from memory_agent import MemoryAgent

def previous_chats(memory_agent: MemoryAgent):
    st.title("Previous Chatrooms")
    
    all_conversations = memory_agent.get_all_conversations()
    
    if not all_conversations:
        st.write("No previous conversations found.")
    else:
        # Create a selectbox with chatroom names
        chatroom_names = list(all_conversations.keys())
        selected_chatroom = st.selectbox("Select a chatroom to view:", chatroom_names)
        
        if selected_chatroom:
            st.write(f"**Chatroom {selected_chatroom}:**")
            for exchange in all_conversations[selected_chatroom]:
                st.write(f"**You:** {exchange['question']}")
                st.write(f"**Response:** {exchange['response']}")
            st.write("---")
    
    # Button to clear previous chats
    if st.button("Clear Previous Chats"):
        memory_agent.clear_conversations()
        st.success("Previous chats cleared!")
        st.rerun()

    # Button to go back to the main chatroom
    if st.button("Back to Chatroom"):
        st.query_params["tab"] = "Chatroom"
        st.rerun()