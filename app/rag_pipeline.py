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
    Convert text chunks into embeddings and store them in FAISS.
    
    Important: Filter out empty chunks first—these typically occur when PDF 
    pages are image-only and text cannot be extracted. 
    Empty chunks cause an IndexError in FAISS.
    """
    
    # Guard: remove empty or very small chunks
    # Chunks shorter than 20 characters are usually not meaningful
    valid_chunks = [
        chunk for chunk in chunks 
        if chunk.page_content and len(chunk.page_content.strip()) > 20
    ]
    
    # If all chunks are empty, raise a meaningful error
    if not valid_chunks:
        raise ValueError(
            "No readable text found in this PDF. "
            "The document may be image-based or scanned. "
            "Please try a text-based PDF."
        )
    
    # Count how many chunks were filtered useful for debugging
    filtered_count = len(chunks) - len(valid_chunks)
    if filtered_count > 0:
        print(f"Note: {filtered_count} empty chunks filtered out of {len(chunks)} total.")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(valid_chunks, embeddings)
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