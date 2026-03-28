# rag_pipeline.py
# Core logic for the RAG pipeline using Groq and HuggingFace Embeddings.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Free alternative to OpenAI Embeddings
from langchain_groq import ChatGroq # Groq integration
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF document and splits it into smaller chunks for efficient processing.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Converts text chunks into vector embeddings using a free HuggingFace model.
    Note: We use HuggingFaceEmbeddings because Groq doesn't provide an embedding endpoint yet.
    """
    # 'all-MiniLM-L6-v2' is small, fast, and runs locally on your CPU for free.
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """
    Sets up the Retrieval Chain using Groq's Llama model for fast generation.
    """
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", # Latest, fast and high-performing model
        temperature=0,
        # Ensure GROQ_API_KEY is set in your environment variables
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the provided context to answer the question. "
        "If the answer is not in the context, say that you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return qa_chain