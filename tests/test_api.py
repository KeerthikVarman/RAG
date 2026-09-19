import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_rag_components():
    with patch("api.VectorStore") as mock_vs, \
         patch("api.Embedding") as mock_emb, \
         patch("api.RAGRetriever") as mock_ret:
        
        vs_instance = MagicMock()
        vs_instance.collection.count.return_value = 5
        mock_vs.return_value = vs_instance
        
        emb_instance = MagicMock()
        mock_emb.return_value = emb_instance
        
        ret_instance = MagicMock()
        mock_ret.return_value = ret_instance

        yield {
            "vector_store": vs_instance,
            "embedding": emb_instance,
            "retriever": ret_instance,
        }


def test_root_endpoint(mock_rag_components):
    from api import app
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Welcome to the RAG PDF AI Assistant API" in data["message"]
        assert data["docs_url"] == "/docs"


def test_health_endpoint(mock_rag_components):
    from api import app
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["indexed_chunks"] == 5


def test_query_endpoint(mock_rag_components):
    from api import app
    mock_sources = [
        {
            "id": "chunk_1",
            "document": "Test document content",
            "metadata": {"source_file": "sample.pdf", "page": 1},
            "distance": 0.1234,
            "rank": 1,
        }
    ]
    with patch("api.rag_with_sources", return_value=("This is the answer.", mock_sources)):
        with TestClient(app) as client:
            response = client.post("/query", json={"query": "What is RAG?", "top_k": 3})
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "This is the answer."
            assert len(data["sources"]) == 1
            assert data["sources"][0]["metadata"]["source_file"] == "sample.pdf"


def test_list_documents_endpoint(tmp_path):
    from api import app
    with patch("api.PDF_DIR", tmp_path):
        sample_pdf = tmp_path / "test.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4 test data")
        
        with TestClient(app) as client:
            response = client.get("/documents")
            assert response.status_code == 200
            data = response.json()
            assert data["total_files"] == 1
            assert data["documents"][0]["filename"] == "test.pdf"


def test_force_reindex_endpoint_empty(tmp_path):
    from api import app
    with patch("api.PDF_DIR", tmp_path), patch("api.load_pdfs", return_value=[]):
        with TestClient(app) as client:
            response = client.post("/index")
            assert response.status_code == 200
            data = response.json()
            assert data["indexed_chunks"] == 0
            assert "No PDF files found" in data["message"]
