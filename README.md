# 🔍 RAG Document Question Answering System

> Ask questions about any PDF using natural language — 
> powered by LangChain, FAISS, and LLMs.

## Live Demo
🚀 [Try it here](https://rag-document-app-zee.streamlit.app)

## What It Does
Upload any PDF document and ask questions in plain English.
The system uses Retrieval-Augmented Generation to find 
relevant sections and generate accurate answers.

An intelligent PDF assistant that uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers from your documents. 

## 🌟 Features
- **Lightning Fast:** Powered by Groq (Llama-3.3-70B model).
- **Free Embeddings:** Uses HuggingFace `all-MiniLM-L6-v2` (runs locally).
- **Smart Retrieval:** Uses FAISS for efficient similarity search.
- **Modern UI:** Clean and interactive Streamlit interface.

## 🛠️ Tech Stack
- **Orchestration:** LangChain 0.3
- **LLM:** Groq (Llama 3.3)
- **Vector DB:** FAISS
- **Embeddings:** HuggingFace Transformers
- **Frontend:** Streamlit

## How It Works
1. PDF is loaded and split into overlapping chunks
2. Each chunk is converted to embeddings (numerical vectors)
3. Embeddings are stored in a FAISS vector database
4. User question is embedded and matched against chunks
5. Top matching chunks + question are sent to LLM
6. LLM generates a contextual answer

## Run Locally
```bash
git clone https://github.com/dev-mzeeshan/rag-document-qa
cd rag-document-qa
pip install -r requirements.txt
cp .env.example .env  # Add your API key
streamlit run app/main.py
```

## Author
**Muhammad Zeeshan** 
AI Engineer  
[Portfolio](https://zeeshan-portfolio-amber.vercel.app) · 
[LinkedIn](https://linkedin.com/in/zeeshanofficial)