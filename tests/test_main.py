"""Unit tests for main FastAPI application"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_embedding_model_instance():
    """Mock EmbeddingModel instance for testing"""
    mock_model = MagicMock()
    mock_model.encode_single.return_value = np.random.rand(384)
    mock_model.get_dimension.return_value = 384
    return mock_model


@pytest.fixture
def mock_qdrant_manager_instance():
    """Mock QdrantManager instance for testing"""
    mock_manager = MagicMock()

    # Mock get_collection_info
    mock_manager.get_collection_info.return_value = {
        'name': 'fables',
        'points_count': 100,
        'vectors_count': 100,
        'status': 'green'
    }

    # Mock search
    mock_manager.search.return_value = [
        {
            'id': 1,
            'score': 0.95,
            'payload': {
                'title': 'The Boy Who Cried Wolf',
                'content': 'A shepherd boy got bored.',
                'moral': 'Liars are not believed.',
                'language': 'en',
                'word_count': 10
            }
        }
    ]

    # Mock retrieve
    mock_point = MagicMock()
    mock_point.id = 1
    mock_point.payload = {
        'title': 'The Boy Who Cried Wolf',
        'content': 'A shepherd boy got bored.',
        'moral': 'Liars are not believed.',
        'language': 'en',
        'word_count': 10
    }
    mock_manager.client.retrieve.return_value = [mock_point]

    return mock_manager


class TestLifecycleEvents:
    """Test application lifecycle events"""

    @pytest.mark.asyncio
    @patch('src.main.EmbeddingModel')
    @patch('src.main.QdrantManager')
    async def test_startup_event_success(self, mock_qdrant_cls, mock_emb_cls):
        """Test successful startup event"""
        # Arrange
        mock_emb = MagicMock()
        mock_emb.get_dimension.return_value = 384
        mock_emb_cls.return_value = mock_emb

        mock_qdrant = MagicMock()
        mock_qdrant.get_collection_info.return_value = {
            'points_count': 100,
            'status': 'green'
        }
        mock_qdrant_cls.return_value = mock_qdrant

        # Import and run startup event
        from src.main import startup_event
        await startup_event()

        # Assert
        mock_emb_cls.assert_called_once()
        mock_qdrant_cls.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.main.EmbeddingModel')
    @patch('src.main.QdrantManager')
    async def test_startup_event_collection_missing(self, mock_qdrant_cls, mock_emb_cls):
        """Test startup when collection doesn't exist"""
        # Arrange
        mock_emb_cls.return_value = MagicMock()

        mock_qdrant = MagicMock()
        mock_qdrant.get_collection_info.return_value = None
        mock_qdrant_cls.return_value = mock_qdrant

        # Import and run startup event
        from src.main import startup_event
        await startup_event()

        # Assert - should complete without errors
        mock_qdrant.get_collection_info.assert_called_once()


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message"""
        # Act
        response = client.get("/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert data["docs"] == "/docs"


class TestHealthCheckEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, client, mock_embedding_model_instance, mock_qdrant_manager_instance):
        """Test health check with initialized system"""
        # Arrange - set global variables
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance
        main_module.qdrant_manager = mock_qdrant_manager_instance

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['total_fables'] == 100
        assert data['collection_name'] == 'fables'

    def test_health_check_not_initialized(self, client):
        """Test health check when system is not initialized"""
        # Arrange - ensure globals are None
        import src.main as main_module
        main_module.embedding_model = None
        main_module.qdrant_manager = None

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 503
        assert "not initialized" in response.json()['detail']

    def test_health_check_collection_missing(self, client, mock_embedding_model_instance):
        """Test health check when collection doesn't exist"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance

        mock_qdrant = MagicMock()
        mock_qdrant.get_collection_info.return_value = None
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 503
        assert "does not exist" in response.json()['detail']

    def test_health_check_exception(self, client, mock_embedding_model_instance):
        """Test health check with exception"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance

        mock_qdrant = MagicMock()
        mock_qdrant.get_collection_info.side_effect = Exception('Database error')
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 503


class TestSearchEndpoint:
    """Test search endpoint"""

    def test_search_success(self, client, mock_embedding_model_instance, mock_qdrant_manager_instance):
        """Test successful search"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance
        main_module.qdrant_manager = mock_qdrant_manager_instance

        # Act
        response = client.post("/search", json={
            "query": "honesty story",
            "limit": 5
        })

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['query'] == 'honesty story'
        assert len(data['results']) >= 1
        assert data['total_results'] >= 1
        assert 'title' in data['results'][0]
        assert 'score' in data['results'][0]

    def test_search_with_threshold(self, client, mock_embedding_model_instance, mock_qdrant_manager_instance):
        """Test search with score threshold"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance
        main_module.qdrant_manager = mock_qdrant_manager_instance

        # Act
        response = client.post("/search", json={
            "query": "honesty story",
            "limit": 5,
            "score_threshold": 0.8
        })

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['total_results'] >= 0

    def test_search_empty_query(self, client):
        """Test search with empty query - should fail validation"""
        # Act
        response = client.post("/search", json={
            "query": "",
            "limit": 5
        })

        # Assert
        assert response.status_code == 422  # Validation error

    def test_search_limit_boundaries(self, client, mock_embedding_model_instance, mock_qdrant_manager_instance):
        """Test search with limit boundaries (1-20)"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance
        main_module.qdrant_manager = mock_qdrant_manager_instance

        # Test valid limit
        response = client.post("/search", json={
            "query": "test",
            "limit": 1
        })
        assert response.status_code == 200

        response = client.post("/search", json={
            "query": "test",
            "limit": 20
        })
        assert response.status_code == 200

        # Test invalid limit (too high)
        response = client.post("/search", json={
            "query": "test",
            "limit": 21
        })
        assert response.status_code == 422

    def test_search_not_initialized(self, client):
        """Test search when system is not initialized"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = None
        main_module.qdrant_manager = None

        # Act
        response = client.post("/search", json={
            "query": "test",
            "limit": 5
        })

        # Assert
        assert response.status_code == 503

    def test_search_no_results(self, client, mock_embedding_model_instance):
        """Test search with no results"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance

        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = []
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.post("/search", json={
            "query": "test",
            "limit": 5
        })

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['total_results'] == 0
        assert len(data['results']) == 0

    def test_search_exception(self, client, mock_embedding_model_instance):
        """Test search with exception"""
        # Arrange
        import src.main as main_module
        main_module.embedding_model = mock_embedding_model_instance

        mock_qdrant = MagicMock()
        mock_qdrant.search.side_effect = Exception('Search error')
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.post("/search", json={
            "query": "test",
            "limit": 5
        })

        # Assert
        assert response.status_code == 500


class TestGetFableEndpoint:
    """Test get fable by ID endpoint"""

    def test_get_fable_by_id_success(self, client, mock_qdrant_manager_instance):
        """Test successfully getting a fable by ID"""
        # Arrange
        import src.main as main_module
        main_module.qdrant_manager = mock_qdrant_manager_instance

        # Act
        response = client.get("/fables/1")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data['id'] == 1
        assert 'title' in data
        assert 'content' in data
        assert 'moral' in data

    def test_get_fable_by_id_not_found(self, client):
        """Test getting a fable that doesn't exist"""
        # Arrange
        import src.main as main_module
        mock_qdrant = MagicMock()
        mock_qdrant.client.retrieve.return_value = []
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.get("/fables/999")

        # Assert
        assert response.status_code == 404

    def test_get_fable_by_id_not_initialized(self, client):
        """Test getting fable when system is not initialized"""
        # Arrange
        import src.main as main_module
        main_module.qdrant_manager = None

        # Act
        response = client.get("/fables/1")

        # Assert
        assert response.status_code == 503

    def test_get_fable_by_id_exception(self, client):
        """Test getting fable with exception"""
        # Arrange
        import src.main as main_module
        mock_qdrant = MagicMock()
        mock_qdrant.client.retrieve.side_effect = Exception('Retrieve error')
        main_module.qdrant_manager = mock_qdrant

        # Act
        response = client.get("/fables/1")

        # Assert
        assert response.status_code == 500
