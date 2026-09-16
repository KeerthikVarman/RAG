import pytest
import numpy as np
from chat import VectorStore


def test_vector_store_initialization(tmp_path):
    """Test VectorStore initializes ChromaDB persistent client in given directory."""
    persist_dir = str(tmp_path / "chroma_db")
    store = VectorStore(collection_name="test_init_collection", persist_directory=persist_dir)

    assert store.collection.name == "test_init_collection"
    assert store.collection.count() == 0


def test_add_documents_to_vector_store(tmp_path, sample_documents):
    """Test adding documents and embeddings to ChromaDB store."""
    persist_dir = str(tmp_path / "chroma_db")
    store = VectorStore(collection_name="test_add_collection", persist_directory=persist_dir)

    # Generate dummy embeddings matching sample_documents count
    dummy_embeddings = np.random.rand(len(sample_documents), 384)

    store.add_documents(sample_documents, dummy_embeddings)

    assert store.collection.count() == len(sample_documents)

    # Verify upserting again does not duplicate items with identical MD5 IDs
    store.add_documents(sample_documents, dummy_embeddings)
    assert store.collection.count() == len(sample_documents)
