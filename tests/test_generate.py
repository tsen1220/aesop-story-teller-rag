"""Unit tests for generate handler"""

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
def mock_embedding_model():
    """Mock EmbeddingModel instance"""
    mock = MagicMock()
    mock.encode_single.return_value = np.random.rand(384)
    return mock


@pytest.fixture
def mock_qdrant_manager():
    """Mock QdrantManager instance"""
    mock = MagicMock()
    mock.search.return_value = [
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
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM provider"""
    mock = MagicMock()
    mock.generate.return_value = "This is a generated response based on the fables."
    return mock


class TestGenerateEndpoint:
    """Test generate endpoint"""

    def test_generate_success(self, client, mock_embedding_model, mock_qdrant_manager, mock_llm):
        """Test successful generation"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager
        deps.llm_providers_cache = {"ollama:llama3.1:8b": mock_llm}

        with patch.object(deps, 'get_llm_provider', return_value=mock_llm):
            response = client.post("/generate", json={
                "query": "What is the moral of the story?",
                "limit": 3
            })

        assert response.status_code == 200
        data = response.json()
        assert data['query'] == "What is the moral of the story?"
        assert 'answer' in data
        assert 'sources' in data
        assert len(data['sources']) >= 1

    def test_generate_with_provider(self, client, mock_embedding_model, mock_qdrant_manager, mock_llm):
        """Test generation with specific provider"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager
        deps.llm_providers_cache = {}

        with patch.object(deps, 'get_llm_provider', return_value=mock_llm):
            response = client.post("/generate", json={
                "query": "Tell me about honesty",
                "limit": 2,
                "provider": "ollama",
                "ollama_model": "llama3.1:8b"
            })

        assert response.status_code == 200

    def test_generate_not_initialized(self, client):
        """Test generate when system not initialized"""
        import src.dependencies as deps
        deps.embedding_model = None
        deps.qdrant_manager = None

        response = client.post("/generate", json={
            "query": "test",
            "limit": 3
        })

        assert response.status_code == 503

    def test_generate_provider_not_available(self, client, mock_embedding_model, mock_qdrant_manager):
        """Test generate with unavailable provider"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager

        response = client.post("/generate", json={
            "query": "test",
            "provider": "nonexistent_provider"
        })

        assert response.status_code == 400
        assert "not available" in response.json()['detail']

    def test_generate_model_not_available(self, client, mock_embedding_model, mock_qdrant_manager):
        """Test generate with unavailable model"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager

        response = client.post("/generate", json={
            "query": "test",
            "provider": "ollama",
            "ollama_model": "nonexistent:model"
        })

        assert response.status_code == 400
        assert "not available" in response.json()['detail']

    def test_generate_llm_init_error(self, client, mock_embedding_model, mock_qdrant_manager):
        """Test generate when LLM initialization fails"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager
        deps.llm_providers_cache = {}

        with patch.object(deps, 'get_llm_provider', side_effect=Exception("Init failed")):
            response = client.post("/generate", json={
                "query": "test",
                "limit": 3
            })

        assert response.status_code == 500
        assert "Failed to initialize" in response.json()['detail']

    def test_generate_llm_returns_none(self, client, mock_embedding_model, mock_qdrant_manager):
        """Test generate when LLM returns None"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager

        mock_llm = MagicMock()
        mock_llm.generate.return_value = None
        deps.llm_providers_cache = {"ollama:llama3.1:8b": mock_llm}

        with patch.object(deps, 'get_llm_provider', return_value=mock_llm):
            response = client.post("/generate", json={
                "query": "test",
                "limit": 3
            })

        assert response.status_code == 500
        assert "failed to generate" in response.json()['detail']

    def test_generate_exception(self, client, mock_embedding_model, mock_qdrant_manager, mock_llm):
        """Test generate with exception during processing"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager
        deps.qdrant_manager.search.side_effect = Exception("Search failed")
        deps.llm_providers_cache = {"ollama:llama3.1:8b": mock_llm}

        with patch.object(deps, 'get_llm_provider', return_value=mock_llm):
            response = client.post("/generate", json={
                "query": "test",
                "limit": 3
            })

        assert response.status_code == 500
        assert "failed" in response.json()['detail'].lower()

    def test_generate_empty_query(self, client):
        """Test generate with empty query"""
        response = client.post("/generate", json={
            "query": "",
            "limit": 3
        })

        assert response.status_code == 422  # Validation error

    def test_generate_limit_boundaries(self, client, mock_embedding_model, mock_qdrant_manager, mock_llm):
        """Test generate with limit boundaries"""
        import src.dependencies as deps
        deps.embedding_model = mock_embedding_model
        deps.qdrant_manager = mock_qdrant_manager
        deps.llm_providers_cache = {"ollama:llama3.1:8b": mock_llm}

        with patch.object(deps, 'get_llm_provider', return_value=mock_llm):
            # Test valid limit
            response = client.post("/generate", json={
                "query": "test",
                "limit": 1
            })
            assert response.status_code == 200

            response = client.post("/generate", json={
                "query": "test",
                "limit": 10
            })
            assert response.status_code == 200

        # Test invalid limit
        response = client.post("/generate", json={
            "query": "test",
            "limit": 11
        })
        assert response.status_code == 422
