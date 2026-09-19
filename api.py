import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from chat import (
    VectorStore,
    Embedding,
    RAGRetriever,
    rag_with_sources,
    load_pdfs,
    split_documents,
    llm,
)

PDF_DIR = Path("data/pdf")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Global instances initialized during application startup
vector_store: Optional[VectorStore] = None
embedding_model: Optional[Embedding] = None
retriever: Optional[RAGRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, embedding_model, retriever
    load_dotenv()
    print("Initializing RAG components for FastAPI server...")
    vector_store = VectorStore()
    embedding_model = Embedding()
    retriever = RAGRetriever(vector_store, embedding_model)
    
    # Auto-index if database is empty but PDFs exist
    if vector_store.collection.count() == 0:
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if pdf_files:
            print("Vector database is empty. Auto-indexing existing PDFs...")
            index_pdf_documents()

    yield
    print("Shutting down RAG FastAPI server.")


app = FastAPI(
    title="RAG PDF AI Assistant API",
    description="FastAPI REST service providing document vector retrieval and Groq LLM answers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- SCHEMAS -----------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to query against uploaded PDFs")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top chunks to retrieve")


class SourceItem(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    rank: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class StatusResponse(BaseModel):
    status: str
    indexed_chunks: int
    groq_api_key_loaded: bool


class IndexResponse(BaseModel):
    indexed_chunks: int
    message: str


class DocumentItem(BaseModel):
    filename: str
    size_bytes: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total_files: int


# ----------------- HELPER FUNCTIONS -----------------
def index_pdf_documents() -> int:
    """Helper function to process and index PDFs into ChromaDB."""
    if not vector_store or not embedding_model:
        raise HTTPException(status_code=500, detail="RAG components not initialized")

    documents = load_pdfs(str(PDF_DIR))
    if not documents:
        return 0
    chunks = split_documents(documents)
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.generate_embedding(texts)
    vector_store.add_documents(chunks, embeddings)
    return len(chunks)


# ----------------- ENDPOINTS -----------------
@app.get("/", summary="Root Endpoint")
def read_root():
    return {
        "message": "Welcome to the RAG PDF AI Assistant API",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=StatusResponse, summary="API Health and Status")
@app.get("/status", response_model=StatusResponse, summary="API Health and Status")
def get_status():
    groq_key = os.getenv("GROQ_API_KEY")
    chunk_count = vector_store.collection.count() if vector_store else 0
    return StatusResponse(
        status="ok",
        indexed_chunks=chunk_count,
        groq_api_key_loaded=bool(groq_key),
    )


@app.post("/query", response_model=QueryResponse, summary="Query RAG Assistant")
def query_rag(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever is not initialized")
    
    answer, sources = rag_with_sources(
        query=request.query,
        retriever=retriever,
        llm=llm,
        top_k=request.top_k,
    )

    formatted_sources = [
        SourceItem(
            id=src.get("id", ""),
            document=src.get("document", ""),
            metadata=src.get("metadata", {}),
            distance=round(float(src.get("distance", 0.0)), 4),
            rank=int(src.get("rank", 1)),
        )
        for src in sources
    ]

    return QueryResponse(answer=answer, sources=formatted_sources)


@app.get("/documents", response_model=DocumentListResponse, summary="List PDF Documents")
def list_documents():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    docs = [
        DocumentItem(filename=pdf.name, size_bytes=pdf.stat().st_size)
        for pdf in pdf_files
    ]
    return DocumentListResponse(documents=docs, total_files=len(docs))


@app.post("/upload", response_model=IndexResponse, summary="Upload PDFs and Index")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    saved_files = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF file. Only PDF files are supported.",
            )
        file_path = PDF_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file.filename)
    
    chunks_indexed = index_pdf_documents()
    return IndexResponse(
        indexed_chunks=chunks_indexed,
        message=f"Successfully uploaded {len(saved_files)} file(s) and indexed {chunks_indexed} chunk(s).",
    )


@app.post("/index", response_model=IndexResponse, summary="Force Re-Index All PDFs")
def force_reindex():
    chunks_indexed = index_pdf_documents()
    if chunks_indexed == 0:
        return IndexResponse(
            indexed_chunks=0,
            message="No PDF files found in data/pdf/ to index.",
        )
    return IndexResponse(
        indexed_chunks=chunks_indexed,
        message=f"Successfully re-indexed {chunks_indexed} chunks into ChromaDB.",
    )
