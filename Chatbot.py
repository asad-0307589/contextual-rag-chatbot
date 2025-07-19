# Chatbot.py

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Page setup
st.set_page_config(page_title="🤖 Context Aware Chatbot", page_icon="🤖")

# Background image (local or hosted)
background_url = "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=1530&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Custom CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url('{background_url}') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Poppins', sans-serif;
    }}

    /* Add spinning animation to the title */
    @keyframes spin {{
        from {{ transform: rotateY(0deg); }}
        to {{ transform: rotateY(360deg); }}
    }}

    .title {{
        font-size: 36px;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
        animation: spin 6s linear infinite;
    }}

    .creator {{
        font-size: 14px;
        color: #cccccc;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
    }}

    .message {{
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 8px;
        padding: 10px;
        color: #ffffff;
        margin-bottom: 8px;
    }}

    /* Input box style */
    .stTextInput > div > div > input {{
        background-color: rgba(0,0,0,0.8);
        color: #ffffff;
        border: 1px solid #888;
        border-radius: 5px;
    }}

    /* Make the input label text white */
    label {{
        color: #ffffff !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Title & creator
st.markdown(f"<div class='title'>🤖 Context Aware Chatbot</div>", unsafe_allow_html=True)
st.markdown(f"<div class='creator'>by Asad Shah</div>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    qa_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=qa_pipe)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

qa_chain = load_qa_chain()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
query = st.text_input("💬 Ask your question:")

if query:
    st.session_state.history.append(("You", query))
    with st.spinner("🤔 Thinking..."):
        answer = qa_chain({"query": query})["result"]
    st.session_state.history.append(("Bot", answer))

# Show history
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div class='message'><strong>🧑‍💻 You:</strong> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='message'><strong>🤖 Bot:</strong> {text}</div>", unsafe_allow_html=True)
