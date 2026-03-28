import streamlit as st
import time
import json
import requests
import tempfile
import os
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
from rag_pipeline import load_and_split_pdf, create_vector_store, create_qa_chain

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG Pro | Zeeshan AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .stChatInputContainer { padding-bottom: 20px; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e313d,#0e1117); }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_49rdyysj.json")
lottie_success = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_at6m7l3i.json")

# --- SIDEBAR: SETTINGS & STATS ---
with st.sidebar:
    st.title("🛠️ Configuration")
    selected_model = st.selectbox("Select Model", 
                                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], 
                                index=0)
    
    st.divider()
    st.subheader("Fine-tuning")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    if "chunk_count" in st.session_state:
        st.metric("Total Chunks", st.session_state.chunk_count)
        st.caption(f"File: {st.session_state.file_name}")

# --- MAIN INTERFACE ---
st.title("RAG Insight Pro")
st.markdown("##### Transform your PDFs into interactive knowledge bases.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# File Upload Section (with Animation)
uploaded_file = st.file_uploader("Drop your PDF here", type="pdf", label_visibility="collapsed")

if uploaded_file:
    if "qa_chain" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.status("🏗️ Building Knowledge Base...", expanded=True) as status:
            st.write("Reading PDF content...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                st.write("Splitting into semantic chunks...")
                chunks = load_and_split_pdf(tmp_path)
                
                st.write("Generating Vector Embeddings (HuggingFace)...")
                vector_store = create_vector_store(chunks)
                
                st.write("Connecting to Groq Engine...")
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                
                status.update(label="✅ Document Ready!", state="complete", expanded=False)
                st.toast("Document processed successfully!", icon="🔥")
            finally:
                if os.path.exists(tmp_path): os.unlink(tmp_path)

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                response = st.session_state.qa_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
                # Show sources in a clean expander
                with st.expander("📚 View Reference Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.caption(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):")
                        st.write(doc.page_content[:300] + "...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload a document first!")