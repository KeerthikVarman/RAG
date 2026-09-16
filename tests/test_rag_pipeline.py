import pytest
from unittest.mock import MagicMock
from chat import rag_with_sources, rag_simple


def test_rag_with_sources_no_context():
    """Test rag_with_sources when no relevant chunks are retrieved."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_llm = MagicMock()

    answer, sources = rag_with_sources("Unknown question", mock_retriever, mock_llm)

    assert answer == "No relevant context found in documents."
    assert sources == []
    mock_llm.invoke.assert_not_called()


def test_rag_with_sources_with_context(mock_retriever, mock_llm):
    """Test rag_with_sources formats context and invokes LLM."""
    answer, sources = rag_with_sources("What is RAG?", mock_retriever, mock_llm, top_k=3)

    assert answer == "This is a mocked LLM response."
    assert len(sources) == 1
    mock_llm.invoke.assert_called_once()

    # Check prompt passed to LLM includes context
    prompt_arg = mock_llm.invoke.call_args[0][0]
    assert "Retrieval-Augmented Generation combines retrieval with generative LLMs." in prompt_arg
    assert "Source: rag_overview.pdf" in prompt_arg


def test_rag_simple(mock_retriever, mock_llm):
    """Test rag_simple helper function returns response text directly."""
    answer = rag_simple("What is RAG?", mock_retriever, mock_llm, top_k=3)
    assert answer == "This is a mocked LLM response."
