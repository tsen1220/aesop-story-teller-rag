"""Unit tests for init_database module"""

import pytest
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from src.init_database import init_fables_collection


class TestInitDatabase:
    """Test database initialization"""

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed fable data"""
        return [
            {
                "id": "fable_01",
                "title": "Test Fable 1",
                "content": "Test content 1",
                "moral": "Test moral 1",
                "language": "en",
                "metadata": {
                    "number": "01",
                    "word_count": 10
                }
            },
            {
                "id": "fable_02",
                "title": "Test Fable 2",
                "content": "Test content 2",
                "moral": "Test moral 2",
                "language": "en",
                "metadata": {
                    "number": "02",
                    "word_count": 15
                }
            }
        ]

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.init_database.json.load')
    def test_init_fables_collection_success(
        self,
        mock_json_load,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls,
        sample_processed_data
    ):
        """Test successful database initialization"""
        # Arrange
        mock_json_load.return_value = sample_processed_data

        # Mock EmbeddingModel
        mock_embedding = MagicMock()
        mock_embedding.get_dimension.return_value = 384
        mock_embedding.encode.return_value = np.random.rand(2, 384)
        mock_embedding.encode_single.return_value = np.random.rand(384)
        mock_embedding_cls.return_value = mock_embedding

        # Mock QdrantManager
        mock_qdrant = MagicMock()
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant.insert_vectors.return_value = True
        mock_qdrant.get_collection_info.return_value = {
            'name': 'fables',
            'vectors_count': 2,
            'points_count': 2,
            'status': 'green'
        }
        mock_qdrant.search.return_value = [
            {
                'payload': {
                    'title': 'Test Fable 1',
                    'moral': 'Test moral 1'
                },
                'score': 0.95
            }
        ]
        mock_qdrant_cls.return_value = mock_qdrant

        # Act
        init_fables_collection()

        # Assert
        mock_embedding.encode.assert_called_once()
        mock_qdrant.delete_collection.assert_called_once()
        mock_qdrant.create_collection.assert_called_once()
        mock_qdrant.insert_vectors.assert_called_once()
        mock_qdrant.get_collection_info.assert_called_once()

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.init_database.json.load')
    def test_init_fables_collection_with_test_search(
        self,
        mock_json_load,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls,
        sample_processed_data
    ):
        """Test initialization includes test search"""
        # Arrange
        mock_json_load.return_value = sample_processed_data

        mock_embedding = MagicMock()
        mock_embedding.get_dimension.return_value = 384
        mock_embedding.encode.return_value = np.random.rand(2, 384)
        mock_embedding.encode_single.return_value = np.random.rand(384)
        mock_embedding_cls.return_value = mock_embedding

        mock_qdrant = MagicMock()
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant.insert_vectors.return_value = True
        mock_qdrant.get_collection_info.return_value = {
            'name': 'fables',
            'vectors_count': 2,
            'points_count': 2,
            'status': 'green'
        }
        mock_qdrant.search.return_value = [
            {
                'payload': {'title': 'Test', 'moral': 'Test moral'},
                'score': 0.95
            }
        ]
        mock_qdrant_cls.return_value = mock_qdrant

        # Act
        init_fables_collection()

        # Assert - verify search was called
        mock_qdrant.search.assert_called_once()
        mock_embedding.encode_single.assert_called_once()

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', side_effect=FileNotFoundError('Data file not found'))
    def test_init_fables_collection_file_not_found(
        self,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls
    ):
        """Test initialization when data file doesn't exist"""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            init_fables_collection()

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.init_database.json.load')
    def test_init_fables_collection_empty_data(
        self,
        mock_json_load,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls
    ):
        """Test initialization with empty data"""
        # Arrange
        mock_json_load.return_value = []

        mock_embedding = MagicMock()
        mock_embedding.get_dimension.return_value = 384
        mock_embedding.encode.return_value = np.array([]).reshape(0, 384)
        mock_embedding_cls.return_value = mock_embedding

        mock_qdrant = MagicMock()
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant.insert_vectors.return_value = True
        mock_qdrant.get_collection_info.return_value = {
            'name': 'fables',
            'vectors_count': 0,
            'points_count': 0,
            'status': 'green'
        }
        mock_qdrant.search.return_value = []
        mock_qdrant_cls.return_value = mock_qdrant

        # Act
        init_fables_collection()

        # Assert
        mock_embedding.encode.assert_called_once_with([], show_progress=True)

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.init_database.json.load')
    def test_init_fables_collection_embedding_failure(
        self,
        mock_json_load,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls,
        sample_processed_data
    ):
        """Test initialization when embedding generation fails"""
        # Arrange
        mock_json_load.return_value = sample_processed_data

        mock_embedding = MagicMock()
        mock_embedding.get_dimension.return_value = 384
        mock_embedding.encode.side_effect = Exception('Embedding failed')
        mock_embedding_cls.return_value = mock_embedding

        mock_qdrant = MagicMock()
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant_cls.return_value = mock_qdrant

        # Act & Assert
        with pytest.raises(Exception, match='Embedding failed'):
            init_fables_collection()

    @patch('src.init_database.QdrantManager')
    @patch('src.init_database.EmbeddingModel')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.init_database.json.load')
    def test_init_fables_collection_insert_failure(
        self,
        mock_json_load,
        mock_file,
        mock_embedding_cls,
        mock_qdrant_cls,
        sample_processed_data
    ):
        """Test initialization when data insertion fails"""
        # Arrange
        mock_json_load.return_value = sample_processed_data

        mock_embedding = MagicMock()
        mock_embedding.get_dimension.return_value = 384
        mock_embedding.encode.return_value = np.random.rand(2, 384)
        mock_embedding.encode_single.return_value = np.random.rand(384)
        mock_embedding_cls.return_value = mock_embedding

        mock_qdrant = MagicMock()
        mock_qdrant.delete_collection.return_value = True
        mock_qdrant.create_collection.return_value = True
        mock_qdrant.insert_vectors.return_value = False  # Insert fails
        mock_qdrant_cls.return_value = mock_qdrant

        # Act
        init_fables_collection()

        # Assert - function should complete even if insert fails
        mock_qdrant.insert_vectors.assert_called_once()
        # get_collection_info should not be called since insert failed
        mock_qdrant.get_collection_info.assert_not_called()
