import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_app_module_imports():
    """Verify app.py imports core RAG components without syntax/runtime issues."""
    import app
    assert hasattr(app, "index_documents")
    assert hasattr(app, "PDF_DIR")


def test_index_documents_no_documents(tmp_path):
    """Test index_documents when PDF directory is empty."""
    with patch("app.PDF_DIR", tmp_path), \
         patch("app.load_pdfs", return_value=[]), \
         patch("streamlit.error") as mock_st_error:

        import app
        result = app.index_documents()

        assert result == 0
        mock_st_error.assert_called_once_with(f"No PDF files found in data/pdf/")


def test_index_documents_with_documents(tmp_path):
    """Test index_documents workflow when PDFs exist and are indexed."""
    mock_docs = [MagicMock()]
    mock_chunks = [MagicMock()]
    mock_chunks[0].page_content = "Chunk 1"

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [[0.1, 0.2]]

    mock_vector_store = MagicMock()

    with patch("app.PDF_DIR", tmp_path), \
         patch("app.load_pdfs", return_value=mock_docs), \
         patch("app.split_documents", return_value=mock_chunks), \
         patch("app.embedding_model", mock_embedding), \
         patch("app.vector_store", mock_vector_store), \
         patch("streamlit.spinner"), \
         patch("streamlit.success"):

        import app
        result = app.index_documents()

        assert result == 1
        mock_vector_store.add_documents.assert_called_once()
