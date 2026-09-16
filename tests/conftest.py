import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

# Set default GROQ_API_KEY for CI environment before importing project modules
os.environ.setdefault("GROQ_API_KEY", "gsk_dummy_key_for_testing")

# Ensure parent directory (project root) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_documents():
    """Fixture providing sample LangChain Document objects."""
    return [
        Document(
            page_content="Artificial Intelligence (AI) is transforming industries.",
            metadata={"source_file": "ai_intro.pdf", "page": 1, "file_type": "pdf"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines retrieval with generative LLMs.",
            metadata={"source_file": "rag_overview.pdf", "page": 1, "file_type": "pdf"}
        ),
        Document(
            page_content="Vector databases store embeddings for efficient similarity search.",
            metadata={"source_file": "vector_db.pdf", "page": 2, "file_type": "pdf"}
        ),
    ]


@pytest.fixture
def mock_llm():
    """Fixture providing a mock ChatGroq LLM instance."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a mocked LLM response."
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def mock_retriever():
    """Fixture providing a mock RAGRetriever instance."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        {
            "id": "doc_id_1",
            "document": "Retrieval-Augmented Generation combines retrieval with generative LLMs.",
            "metadata": {"source_file": "rag_overview.pdf", "page": 1},
            "distance": 0.05,
            "rank": 1,
        }
    ]
    return retriever
