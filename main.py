import streamlit as st
import ollama
from typing import Dict, Generator

# Function to create a generator for Ollama server output
def ollama_generator(model_name: str, messages: Dict) -> Generator:
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk['message']['content']

# Streamlit application layout
st.title("Ollama with Streamlit Demo")  # Webpage title

# Save webpage state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a model selection dropdown (pulls models from local Ollama server)
st.session_state.selected_model = st.selectbox(
    "Please select the model:", [model["name"] for model in ollama.list()["models"]])

# Display user messages and save model output
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How could I help you?"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.empty()  # Placeholder for the response
        response_text = ""
        for chunk in ollama_generator(st.session_state.selected_model, st.session_state.messages):
            response_text += chunk
            response.markdown(response_text)
        
    # Save model output
    st.session_state.messages.append({"role": "assistant", "content": response_text})
