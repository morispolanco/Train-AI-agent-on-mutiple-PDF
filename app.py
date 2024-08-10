import streamlit as st
import requests
import json
from PyPDF2 import PdfReader

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Función para enviar una solicitud a la API de Together
def train_model_with_together(text):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['together_api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 2512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["\"\""],
        "stream": True
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

# Interfaz de Streamlit
st.title("Entrenamiento LLM con PDFs usando Together API")

# Subir archivos PDF
uploaded_files = st.file_uploader("Sube uno o más archivos PDF", accept_multiple_files=True, type=["pdf"])

if st.button("Entrenar Modelo"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Extrayendo texto de {uploaded_file.name}..."):
                text = extract_text_from_pdf(uploaded_file)

            with st.spinner(f"Entrenando modelo con {uploaded_file.name}..."):
                result = train_model_with_together(text)

            st.success(f"Entrenamiento completado para {uploaded_file.name}!")
            st.json(result)
    else:
        st.warning("Por favor sube al menos un archivo PDF.")
