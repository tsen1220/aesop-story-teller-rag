"""Unit tests for qdrant_manager module"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from qdrant_client.models import Distance
from src.qdrant_manager import QdrantManager


class TestQdrantManager:
    """Test QdrantManager class"""

    @patch('src.qdrant_manager.QdrantClient')
    def test_init_default_connection(self, mock_client):
        """Test initialization with default connection parameters"""
        # Arrange
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Act
        manager = QdrantManager()

        # Assert
        mock_client.assert_called_once_with(host='localhost', port=6333)
        assert manager.client == mock_instance

    @patch('src.qdrant_manager.QdrantClient')
    @patch.dict('os.environ', {'QDRANT_HOST': 'test-host', 'QDRANT_PORT': '9999'})
    def test_init_custom_connection(self, mock_client):
        """Test initialization with custom host and port from environment"""
        # Arrange
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        # Act
        manager = QdrantManager()

        # Assert
        mock_client.assert_called_once_with(host='test-host', port=9999)
        assert manager.client == mock_instance

    @patch('src.qdrant_manager.QdrantClient')
    def test_create_collection_success(self, mock_client):
        """Test successful collection creation"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.get_collections.return_value.collections = []
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        result = manager.create_collection('test_collection', vector_size=384)

        # Assert
        assert result is True
        mock_instance.create_collection.assert_called_once()
        call_kwargs = mock_instance.create_collection.call_args[1]
        assert call_kwargs['collection_name'] == 'test_collection'

    @patch('src.qdrant_manager.QdrantClient')
    def test_create_collection_already_exists(self, mock_client):
        """Test creating a collection that already exists"""
        # Arrange
        mock_instance = MagicMock()

        # Mock existing collection
        mock_collection = MagicMock()
        mock_collection.name = 'test_collection'
        mock_instance.get_collections.return_value.collections = [mock_collection]
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        result = manager.create_collection('test_collection', vector_size=384)

        # Assert
        assert result is False
        mock_instance.create_collection.assert_not_called()

    @patch('src.qdrant_manager.QdrantClient')
    def test_create_collection_exception(self, mock_client):
        """Test collection creation with exception"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.get_collections.side_effect = Exception('Connection error')
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        result = manager.create_collection('test_collection', vector_size=384)

        # Assert
        assert result is False

    @patch('src.qdrant_manager.QdrantClient')
    def test_delete_collection_success(self, mock_client):
        """Test successful collection deletion"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.delete_collection.return_value = True
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        result = manager.delete_collection('test_collection')

        # Assert
        assert result is True
        mock_instance.delete_collection.assert_called_once_with('test_collection')

    @patch('src.qdrant_manager.QdrantClient')
    def test_delete_collection_exception(self, mock_client):
        """Test collection deletion with exception"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.delete_collection.side_effect = Exception('Delete failed')
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        result = manager.delete_collection('test_collection')

        # Assert
        assert result is False

    @patch('src.qdrant_manager.QdrantClient')
    def test_insert_vectors_with_ids(self, mock_client):
        """Test vector insertion with provided IDs"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.upsert.return_value = True
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        vectors = [np.random.rand(384) for _ in range(3)]
        payloads = [{'title': f'test{i}'} for i in range(3)]
        ids = ['id1', 'id2', 'id3']

        # Act
        result = manager.insert_vectors('test_collection', vectors, payloads, ids)

        # Assert
        assert result is True
        mock_instance.upsert.assert_called_once()
        call_kwargs = mock_instance.upsert.call_args[1]
        assert call_kwargs['collection_name'] == 'test_collection'
        assert len(call_kwargs['points']) == 3

    @patch('src.qdrant_manager.QdrantClient')
    @patch('src.qdrant_manager.uuid.uuid4')
    def test_insert_vectors_auto_generate_ids(self, mock_uuid, mock_client):
        """Test vector insertion with auto-generated IDs"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.upsert.return_value = True
        mock_client.return_value = mock_instance

        # Mock UUID generation
        mock_uuid.side_effect = ['uuid1', 'uuid2', 'uuid3']

        manager = QdrantManager()
        vectors = [np.random.rand(384) for _ in range(3)]
        payloads = [{'title': f'test{i}'} for i in range(3)]

        # Act
        result = manager.insert_vectors('test_collection', vectors, payloads)

        # Assert
        assert result is True
        assert mock_uuid.call_count == 3
        mock_instance.upsert.assert_called_once()

    @patch('src.qdrant_manager.QdrantClient')
    def test_insert_vectors_numpy_array(self, mock_client):
        """Test vector insertion with numpy arrays (tests .tolist() conversion)"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.upsert.return_value = True
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        vectors = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        payloads = [{'title': 'test1'}, {'title': 'test2'}]
        ids = ['id1', 'id2']

        # Act
        result = manager.insert_vectors('test_collection', vectors, payloads, ids)

        # Assert
        assert result is True
        mock_instance.upsert.assert_called_once()

    @patch('src.qdrant_manager.QdrantClient')
    def test_insert_vectors_exception(self, mock_client):
        """Test vector insertion with exception"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.upsert.side_effect = Exception('Insert failed')
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        vectors = [np.random.rand(384)]
        payloads = [{'title': 'test'}]

        # Act
        result = manager.insert_vectors('test_collection', vectors, payloads)

        # Assert
        assert result is False

    @patch('src.qdrant_manager.QdrantClient')
    def test_search_success(self, mock_client):
        """Test successful vector search"""
        # Arrange
        mock_instance = MagicMock()

        # Mock search results
        mock_result1 = MagicMock()
        mock_result1.id = 1
        mock_result1.score = 0.95
        mock_result1.payload = {'title': 'Test Fable 1'}

        mock_result2 = MagicMock()
        mock_result2.id = 2
        mock_result2.score = 0.88
        mock_result2.payload = {'title': 'Test Fable 2'}

        mock_instance.search.return_value = [mock_result1, mock_result2]
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        query_vector = [0.1] * 384

        # Act
        results = manager.search('test_collection', query_vector, limit=5)

        # Assert
        assert len(results) == 2
        assert results[0]['id'] == 1
        assert results[0]['score'] == 0.95
        assert results[0]['payload']['title'] == 'Test Fable 1'
        mock_instance.search.assert_called_once_with(
            collection_name='test_collection',
            query_vector=query_vector,
            limit=5,
            score_threshold=None
        )

    @patch('src.qdrant_manager.QdrantClient')
    def test_search_with_threshold(self, mock_client):
        """Test vector search with score threshold"""
        # Arrange
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {'title': 'Test Fable'}
        mock_instance.search.return_value = [mock_result]
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        query_vector = [0.1] * 384

        # Act
        results = manager.search('test_collection', query_vector, limit=5, score_threshold=0.8)

        # Assert
        assert len(results) == 1
        mock_instance.search.assert_called_once_with(
            collection_name='test_collection',
            query_vector=query_vector,
            limit=5,
            score_threshold=0.8
        )

    @patch('src.qdrant_manager.QdrantClient')
    def test_search_exception(self, mock_client):
        """Test search with exception"""
        # Arrange
        mock_instance = MagicMock()
        mock_instance.search.side_effect = Exception('Search failed')
        mock_client.return_value = mock_instance

        manager = QdrantManager()
        query_vector = [0.1] * 384

        # Act
        results = manager.search('test_collection', query_vector)

        # Assert
        assert results == []

    @patch('src.qdrant_manager.QdrantClient')
    def test_get_collection_info(self, mock_client):
        """Test getting collection information"""
        # Arrange
        mock_instance = MagicMock()
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = 'green'
        mock_instance.get_collection.return_value = mock_info
        mock_client.return_value = mock_instance

        manager = QdrantManager()

        # Act
        info = manager.get_collection_info('test_collection')

        # Assert
        assert info is not None
        assert info['name'] == 'test_collection'
        assert info['vectors_count'] == 100
        assert info['points_count'] == 100
        assert info['status'] == 'green'
        mock_instance.get_collection.assert_called_once_with('test_collection')
