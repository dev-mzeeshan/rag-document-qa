# Core logic for the RAG pipeline using Groq and HuggingFace Embeddings.
import os
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_and_split_pdf(pdf_path: str):
    """
    Loads a PDF file and partitions it into manageable text segments.
    
    Args:
        pdf_path (str): The local path to the target PDF document.
        
    Returns:
        list: A list of document objects representing the smaller text chunks.
    """
    # PyPDFium2 is utilized here for its high performance and reliability in PDF parsing.
    loader = PyPDFium2Loader(pdf_path)
    documents = loader.load()
    
    # Recursive splitting ensures that semantic context is preserved by attempting 
    # to split on paragraphs, then sentences, and finally words.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Target character count per chunk
        chunk_overlap=200,   # Overlap to prevent loss of context at the borders
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Transforms text chunks into vector embeddings and stores them in a local index.
    
    Args:
        chunks (list): The list of text segments to be embedded.
        
    Returns:
        FAISS: A searchable vector database instance.
    """
    # Using 'all-MiniLM-L6-v2' provides a cost-effective and efficient way to generate 
    # embeddings locally on the CPU while maintaining high retrieval accuracy.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize FAISS, a robust library for efficient similarity search of dense vectors.
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """
    Configures the full Retrieval-Augmented Generation (RAG) pipeline.
    
    Args:
        vector_store (FAISS): The vector database used for context retrieval.
        
    Returns:
        Chain: A LangChain retrieval chain ready for inference.
    """
    # Initialize the Groq LLM interface.
    # Temperature is set to 0 to prioritize factual accuracy and deterministic outputs.
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
    )

    # Define the assistant's persona and logic for processing the retrieved context.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the provided context to answer the question. "
        "If the answer is not in the context, say that you don't know."
        "\n\n"
        "{context}"
    )
    
    # Construct the chat prompt template for the RAG workflow.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # create_stuff_documents_chain: Combines retrieved documents into the system prompt.
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Set the vector store as a retriever, fetching the top 4 most relevant chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Link the retriever with the document combination chain to form the final pipeline.
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return qa_chain