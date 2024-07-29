import streamlit as st
import ollama
from typing import Dict, Generator

# Define different templates for each chatroom
templates = {
    "簽約過程": (
        "我們提供了以下的背景信息。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "請總結簽約過程中涉及的租賃市場規範、民法義務和權利、以及消費者保護法的主要要點，並說明這些規定如何影響簽約過程。\n"
    ),
    "租約期間": (
        "我們提供了以下的背景信息。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "根據租賃期間內的設備維修責任、租約爭議處理機制和其他可能出現的問題，請總結租賃期間的關鍵點，並提供相應的解決建議。\n"
    ),
    "終止合約": (
        "我們提供了以下的背景信息。 \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "請總結終止合約時需要遵守的法律程序、刑法問題、以及租賃市場規範與公平租賃條例，並說明這些規定如何影響合約的終止過程。\n"
    )
}

# Sidebar options for different chatrooms
chatroom = st.sidebar.selectbox("選擇聊天主題", ("簽約過程", "租約期間", "終止合約"))

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
    "請選擇模型:", [model["name"] for model in ollama.list()["models"]])

# Display user messages and save model output
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("我能幫助你什麼嗎?"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.empty()  # Placeholder for the response
        response_text = ""
        
        # Generate context and query strings using the selected template
        context_str = "你的上下文信息在這裡。"  # Example context, replace with actual context
        query_str = prompt
        template = templates[chatroom]
        formatted_message = template.format(context_str=context_str, query_str=query_str)
        st.session_state.messages.append({"role": "user", "content": formatted_message})
        
        for chunk in ollama_generator(st.session_state.selected_model, st.session_state.messages):
            response_text += chunk
            response.markdown(response_text)
        
    # Save model output
    st.session_state.messages.append({"role": "assistant", "content": response_text})
