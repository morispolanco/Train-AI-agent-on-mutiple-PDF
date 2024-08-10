import streamlit as st
import requests
from dotenv import load_dotenv
import os

def handle_userInput(user_question):
    api_url = "https://api.together.xyz/v1/chat/completions"
    api_key = os.getenv("TOGETHER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "messages": [{"role": "user", "content": user_question}],
        "max_tokens": 2512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stream": False
    }
    response = requests.post(api_url, headers=headers, json=data)
    response_json = response.json()
    return response_json["message"]["content"]

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Agent", page_icon=":robot_face:")

    st.header("Chat Agent :robot_face:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input(label="Pregunta", placeholder="Escribe algo aquí...")
    
    if user_question:
        with st.spinner("Obteniendo respuesta..."):
            response = handle_userInput(user_question)
            st.session_state.chat_history.append({"question": user_question, "response": response})
    
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write(f"**Tú:** {chat['question']}")
            st.write(f"**Agente:** {chat['response']}")

if __name__ == "__main__":
    main()
