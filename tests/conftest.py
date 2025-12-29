"""Shared test fixtures for all test modules"""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_fable_data():
    """Sample fable data for testing"""
    return {
        "id": "fable_01",
        "title": "The Boy Who Cried Wolf",
        "content": "A shepherd boy got bored and cried wolf for fun. When the wolf really came, nobody believed him.",
        "moral": "Liars are not believed even when they speak the truth.",
        "language": "en",
        "metadata": {
            "number": "01",
            "characters": ["boy", "wolf", "villagers"],
            "word_count": 50
        }
    }


@pytest.fixture
def sample_raw_fable():
    """Sample raw fable data (before processing)"""
    return {
        "number": "01",
        "title": "The Boy Who Cried Wolf",
        "story": [
            "A shepherd boy got bored watching his flock.",
            "He cried 'Wolf!' to get attention.",
            "When the wolf really came, nobody believed him."
        ],
        "moral": "Liars are not believed even when they speak the truth.",
        "characters": ["boy", "wolf", "villagers"]
    }


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors (384 dimensions)"""
    return np.random.rand(5, 384).astype(np.float32)


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model"""
    mock_model = MagicMock()

    def mock_encode(texts, show_progress_bar=True, convert_to_numpy=True):
        """Mock encode method that returns appropriate shaped arrays"""
        if isinstance(texts, str):
            return np.random.rand(384).astype(np.float32)
        else:
            return np.random.rand(len(texts), 384).astype(np.float32)

    mock_model.encode.side_effect = mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 384

    return mock_model


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client with common operations"""
    mock_client = MagicMock()

    # Mock collections
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection.return_value = True
    mock_client.delete_collection.return_value = True
    mock_client.upsert.return_value = True

    # Mock search
    def mock_search(*args, **kwargs):
        """Mock search that returns test results"""
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {
            'title': 'The Boy Who Cried Wolf',
            'content': 'A shepherd boy got bored.',
            'moral': 'Liars are not believed.',
            'language': 'en',
            'word_count': 10
        }
        return [mock_result]

    mock_client.search.side_effect = mock_search

    # Mock retrieve
    def mock_retrieve(*args, **kwargs):
        """Mock retrieve that returns points by ID"""
        ids = kwargs.get('ids', [1])
        if not ids:
            return []
        mock_point = MagicMock()
        mock_point.id = ids[0]
        mock_point.payload = {
            'title': 'The Boy Who Cried Wolf',
            'content': 'A shepherd boy got bored.',
            'moral': 'Liars are not believed.',
            'language': 'en',
            'word_count': 10
        }
        return [mock_point]

    mock_client.retrieve.side_effect = mock_retrieve

    # Mock get_collection
    def mock_get_collection(*args, **kwargs):
        """Mock get_collection that returns collection info"""
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = 'green'
        return mock_info

    mock_client.get_collection.side_effect = mock_get_collection

    return mock_client


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing"""
    def _create_json_file(data, filename="test.json"):
        """Helper to create JSON files in temp directory"""
        file_path = tmp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return str(file_path)

    return _create_json_file


@pytest.fixture(autouse=True)
def reset_main_globals():
    """Reset main.py global variables before each test"""
    import src.main as main_module
    main_module.embedding_model = None
    main_module.qdrant_manager = None
    yield
    main_module.embedding_model = None
    main_module.qdrant_manager = None


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    def _set_env(**kwargs):
        """Set multiple environment variables"""
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))

    return _set_env
