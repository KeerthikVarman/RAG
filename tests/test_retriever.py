import pytest
import numpy as np
from unittest.mock import MagicMock
from chat import RAGRetriever


def test_rag_retriever_successful_retrieval():
    """Test RAGRetriever formats query embeddings and ChromaDB results correctly."""
    mock_vector_store = MagicMock()
    mock_vector_store.collection.query.return_value = {
        "ids": [["id_100"]],
        "documents": [["Deep learning models require large datasets."]],
        "metadatas": [[{"source_file": "dl.pdf", "page": 3}]],
        "distances": [[0.1234]],
    }

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.generate_embedding.return_value = np.zeros((1, 384))

    retriever = RAGRetriever(mock_vector_store, mock_embedding_manager)
    results = retriever.retrieve("deep learning", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "id_100"
    assert results[0]["document"] == "Deep learning models require large datasets."
    assert results[0]["metadata"]["source_file"] == "dl.pdf"
    assert results[0]["distance"] == 0.1234
    assert results[0]["rank"] == 1


def test_rag_retriever_handles_exception():
    """Test RAGRetriever catches query exceptions and returns empty list."""
    mock_vector_store = MagicMock()
    mock_vector_store.collection.query.side_effect = Exception("ChromaDB query failure")

    mock_embedding_manager = MagicMock()
    mock_embedding_manager.generate_embedding.return_value = np.zeros((1, 384))

    retriever = RAGRetriever(mock_vector_store, mock_embedding_manager)
    results = retriever.retrieve("broken query", top_k=3)

    assert results == []
