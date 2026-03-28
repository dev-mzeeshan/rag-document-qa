# 🔍 RAG Document Question Answering System

> Ask questions about any PDF using natural language — 
> powered by LangChain, FAISS, and LLMs.

## Live Demo
🚀 [Try it here](https://your-streamlit-link.streamlit.app)

## What It Does
Upload any PDF document and ask questions in plain English.
The system uses Retrieval-Augmented Generation to find 
relevant sections and generate accurate answers.

## Tech Stack
- **LangChain** — LLM orchestration framework
- **FAISS** — Vector similarity search (Facebook AI)
- **OpenAI / Groq** — Language model for answer generation
- **Streamlit** — Web interface
- **PyPDF** — PDF parsing

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
**Muhammad Zeeshan** — AI Engineer  
[Portfolio](https://zeeshan-portfolio-amber.vercel.app) · 
[LinkedIn](https://linkedin.com/in/zeeshanofficial)