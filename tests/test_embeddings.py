"""Unit tests for embeddings module"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Test EmbeddingModel class"""

    @patch('src.embeddings.SentenceTransformer')
    def test_init_default_model(self, mock_transformer):
        """Test initialization with default model"""
        # Arrange
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Act
        embedding_model = EmbeddingModel()

        # Assert
        mock_transformer.assert_called_once_with('paraphrase-multilingual-MiniLM-L12-v2')
        assert embedding_model.dimension == 384
        assert embedding_model.model == mock_model

    @patch('src.embeddings.SentenceTransformer')
    @patch.dict('os.environ', {'EMBEDDING_MODEL': 'test-custom-model'})
    def test_init_custom_model(self, mock_transformer):
        """Test initialization with custom model from environment variable"""
        # Arrange
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Act
        embedding_model = EmbeddingModel()

        # Assert
        mock_transformer.assert_called_once_with('test-custom-model')
        assert embedding_model.dimension == 384
        assert embedding_model.model == mock_model

    @patch('src.embeddings.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_transformer):
        """Test encoding multiple texts"""
        # Arrange
        mock_model = MagicMock()
        expected_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_model.encode.return_value = expected_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()
        texts = ["text1", "text2", "text3"]

        # Act
        result = embedding_model.encode(texts, show_progress=False)

        # Assert
        mock_model.encode.assert_called_once_with(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        np.testing.assert_array_equal(result, expected_embeddings)
        assert result.shape == (3, 384)

    @patch('src.embeddings.SentenceTransformer')
    def test_encode_single_text(self, mock_transformer):
        """Test encoding a single text"""
        # Arrange
        mock_model = MagicMock()
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = expected_embedding
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()
        text = "single text"

        # Act
        result = embedding_model.encode_single(text)

        # Assert
        mock_model.encode.assert_called_once_with(text, convert_to_numpy=True)
        np.testing.assert_array_equal(result, expected_embedding)
        assert result.shape == (384,)

    @patch('src.embeddings.SentenceTransformer')
    def test_encode_empty_list(self, mock_transformer):
        """Test encoding empty list (boundary condition)"""
        # Arrange
        mock_model = MagicMock()
        expected_embeddings = np.array([]).reshape(0, 384).astype(np.float32)
        mock_model.encode.return_value = expected_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()
        texts = []

        # Act
        result = embedding_model.encode(texts, show_progress=False)

        # Assert
        mock_model.encode.assert_called_once_with(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        assert result.shape == (0, 384)

    @patch('src.embeddings.SentenceTransformer')
    def test_encode_single_item_list(self, mock_transformer):
        """Test encoding single-item list (boundary condition)"""
        # Arrange
        mock_model = MagicMock()
        expected_embeddings = np.random.rand(1, 384).astype(np.float32)
        mock_model.encode.return_value = expected_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()
        texts = ["single text"]

        # Act
        result = embedding_model.encode(texts, show_progress=False)

        # Assert
        mock_model.encode.assert_called_once_with(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        assert result.shape == (1, 384)

    @patch('src.embeddings.SentenceTransformer')
    def test_get_dimension(self, mock_transformer):
        """Test getting vector dimension"""
        # Arrange
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()

        # Act
        dimension = embedding_model.get_dimension()

        # Assert
        assert dimension == 384

    @patch('src.embeddings.SentenceTransformer')
    def test_encode_with_progress_disabled(self, mock_transformer):
        """Test encoding with progress bar disabled"""
        # Arrange
        mock_model = MagicMock()
        expected_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_model.encode.return_value = expected_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        embedding_model = EmbeddingModel()
        texts = ["text1", "text2"]

        # Act
        result = embedding_model.encode(texts, show_progress=False)

        # Assert - verify show_progress_bar is False
        mock_model.encode.assert_called_once_with(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        assert result.shape == (2, 384)
