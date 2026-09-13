import os
import sys
import shutil
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# If executed directly with `python app.py`, launch Streamlit CLI automatically
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists
        if not exists():
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
            sys.exit(stcli.main())
    except Exception:
        pass


# Import core RAG components from chat.py
from chat import (
    VectorStore,
    Embedding,
    RAGRetriever,
    rag_with_sources,
    load_pdfs,
    split_documents,
    llm,
)

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .chunk-box {
        background-color: #1E293B;
        border-left: 4px solid #6366F1;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cached resource initialization to prevent reloading model on every rerun
@st.cache_resource(show_spinner="Loading Embedding Model & Vector Store...")
def get_rag_components():
    vector_store = VectorStore()
    embedding_model = Embedding()
    retriever = RAGRetriever(vector_store, embedding_model)
    return vector_store, embedding_model, retriever

vector_store, embedding_model, retriever = get_rag_components()

# PDF directory setup
PDF_DIR = Path("data/pdf")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Helper function to re-index documents
def index_documents():
    with st.spinner("Processing & indexing PDF documents..."):
        documents = load_pdfs(str(PDF_DIR))
        if not documents:
            st.error("No PDF files found in data/pdf/")
            return 0
        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_model.generate_embedding(texts)
        vector_store.add_documents(chunks, embeddings)
        st.success(f"Successfully indexed {len(chunks)} chunks into ChromaDB!")
        return len(chunks)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ RAG Dashboard")
    st.markdown("---")
    
    # Vector Database Status
    chunk_count = vector_store.collection.count()
    st.metric(label="Indexed Chunks in DB", value=chunk_count)

    # API Key status
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✅ GROQ_API_KEY Loaded")
    else:
        st.error("⚠️ GROQ_API_KEY missing in .env")

    st.markdown("---")
    st.subheader("📁 Document Management")

    # Display existing PDF files
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if pdf_files:
        st.write("**Current PDF Files:**")
        for pdf in pdf_files:
            st.markdown(f"- 📄 `{pdf.name}`")
    else:
        st.info("No PDF documents uploaded yet.")

    # Upload new PDF files
    uploaded_files = st.file_uploader(
        "Upload new PDFs", type=["pdf"], accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Save & Index PDFs", use_container_width=True):
            for uploaded_file in uploaded_files:
                file_path = PDF_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
            st.success("Uploaded files saved!")
            index_documents()
            st.rerun()

    if st.button("🔄 Force Re-Index All PDFs", use_container_width=True):
        index_documents()
        st.rerun()

    st.markdown("---")
    st.subheader("🎛️ Settings")
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=3)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-header">📚 RAG PDF Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your uploaded PDF documents with vector retrieval and Groq LLM</div>',
    unsafe_allow_html=True,
)

# Auto-index if database is empty but PDFs exist
if vector_store.collection.count() == 0:
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if pdf_files:
        st.warning("Vector database is empty. Indexing existing PDFs...")
        index_documents()
        st.rerun()
    else:
        st.info("💡 Please upload PDF files in the sidebar to get started!")

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📌 View Source Context & Citations"):
                for src in message["sources"]:
                    meta = src.get("metadata", {})
                    source_file = meta.get("source_file", "Unknown")
                    page = meta.get("page", 0)
                    dist = round(src.get("distance", 0), 4)
                    st.markdown(
                        f"**Source:** `{source_file}` (Page {page}) | **Distance:** `{dist}`"
                    )
                    st.caption(src.get("document", ""))
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            answer, sources = rag_with_sources(
                query=prompt, retriever=retriever, llm=llm, top_k=top_k
            )
            st.markdown(answer)
            
            if sources:
                with st.expander("📌 View Source Context & Citations"):
                    for src in sources:
                        meta = src.get("metadata", {})
                        source_file = meta.get("source_file", "Unknown")
                        page = meta.get("page", 0)
                        dist = round(src.get("distance", 0), 4)
                        st.markdown(
                            f"**Source:** `{source_file}` (Page {page}) | **Distance:** `{dist}`"
                        )
                        st.caption(src.get("document", ""))
                        st.markdown("---")

    # Save to session history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
