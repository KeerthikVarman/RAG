import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document
from chat import load_pdfs, split_documents


def test_load_pdfs_empty_directory(tmp_path):
    """Test load_pdfs returns an empty list when no PDFs exist."""
    docs = load_pdfs(str(tmp_path))
    assert docs == []


def test_load_pdfs_with_mocked_pymupdf(tmp_path):
    """Test load_pdfs successfully processes PDF files and attaches metadata."""
    # Create a dummy pdf file in tmp_path
    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content")

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [
        Document(page_content="Page 1 content", metadata={"page": 1})
    ]

    with patch("chat.PyMuPDFLoader", return_value=mock_loader_instance):
        docs = load_pdfs(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].page_content == "Page 1 content"
    assert docs[0].metadata["source_file"] == "test_doc.pdf"
    assert docs[0].metadata["file_type"] == "pdf"


def test_load_pdfs_handles_exception(tmp_path):
    """Test load_pdfs catches PyMuPDFLoader errors gracefully."""
    pdf_file = tmp_path / "corrupt.pdf"
    pdf_file.write_bytes(b"corrupt content")

    with patch("chat.PyMuPDFLoader", side_effect=Exception("Failed to parse PDF")):
        docs = load_pdfs(str(tmp_path))

    assert docs == []


def test_split_documents(sample_documents):
    """Test split_documents produces expected chunks with RecursiveCharacterTextSplitter."""
    long_doc = [
        Document(
            page_content="Word " * 500,
            metadata={"source_file": "long.pdf", "page": 1}
        )
    ]
    chunks = split_documents(long_doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.metadata["source_file"] == "long.pdf" for chunk in chunks)
