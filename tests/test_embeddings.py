import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from chat import Embedding


def test_embedding_initialization():
    """Test Embedding class initialization with custom model name."""
    mock_st_instance = MagicMock()
    mock_st_instance.get_embedding_dimension.return_value = 384

    with patch("chat.SentenceTransformer", return_value=mock_st_instance) as mock_st_class:
        emb = Embedding(model_name="custom-model")
        mock_st_class.assert_called_once_with("custom-model")
        assert emb.model_name == "custom-model"


def test_generate_embedding_with_mock():
    """Test generate_embedding converts input texts into numpy array embeddings."""
    mock_st_instance = MagicMock()
    fake_embeddings = np.zeros((2, 384))
    mock_st_instance.encode.return_value = fake_embeddings

    with patch("chat.SentenceTransformer", return_value=mock_st_instance):
        emb = Embedding()
        result = emb.generate_embedding(["Hello", "World"])

        mock_st_instance.encode.assert_called_once_with(["Hello", "World"], show_progress_bar=True)
        assert result.shape == (2, 384)


@pytest.mark.integration
def test_real_embedding_generation_shape():
    """Integration test checking actual SentenceTransformer dimension."""
    emb = Embedding(model_name="all-MiniLM-L6-v2")
    texts = ["Testing real sentence transformer"]
    embeddings = emb.generate_embedding(texts)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 384
