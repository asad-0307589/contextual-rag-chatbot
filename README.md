# 🤖 Contextual RAG Chatbot

A fully local and open-source chatbot built with **LangChain**, **FAISS**, and **HuggingFace**, capable of retrieving and answering context-aware questions.  
Built as an experiment to understand how Retrieval-Augmented Generation (RAG) lets chatbots remember and reason over external knowledge.

<img width="1335" height="531" alt="image" src="https://github.com/user-attachments/assets/57eb4cc9-44b1-46ce-bfb6-8d8eb584b4d7" />


---

## ✨ **Project Motivation**
> I always wondered how chatbots can remember context and provide relevant answers.  
> To explore this, I learned about **RAG pipelines** and **LangChain**, and built this project as a practical demonstration.

---

## 🛠 **Tech Stack & Libraries**
- **RAG pipeline**: Retrieval-Augmented Generation
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: google/flan-t5-small (HuggingFace pipeline)
- **Frontend**: Streamlit
- **Framework**: LangChain

---

## 🚀 **Features**
✅ Fully local & offline retrieval  
✅ Context-aware responses  
✅ Clean Streamlit interface with custom styling  
✅ Lightweight, runs on CPU  

---

## 📦 **Installation**
```bash
git clone https://github.com/yourusername/contextual-rag-chatbot.git
cd contextual-rag-chatbot
pip install -r requirements.txt
