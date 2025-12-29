"""Unit tests for dependencies module"""

import pytest
from unittest.mock import patch, MagicMock


class TestGetLLMProvider:
    """Test get_llm_provider factory function"""

    @patch('src.dependencies.Ollama')
    def test_get_ollama_provider(self, mock_ollama):
        """Test getting Ollama provider"""
        from src.dependencies import get_llm_provider

        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        result = get_llm_provider("ollama", "llama3.1:8b")

        mock_ollama.assert_called_once_with(model="llama3.1:8b")
        assert result == mock_instance

    @patch('src.dependencies.GeminiCLI')
    def test_get_gemini_provider(self, mock_gemini):
        """Test getting Gemini CLI provider"""
        from src.dependencies import get_llm_provider

        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        result = get_llm_provider("gemini_cli")

        mock_gemini.assert_called_once()
        assert result == mock_instance

    @patch('src.dependencies.ClaudeCLI')
    def test_get_claude_provider(self, mock_claude):
        """Test getting Claude Code provider"""
        from src.dependencies import get_llm_provider

        mock_instance = MagicMock()
        mock_claude.return_value = mock_instance

        result = get_llm_provider("claude_code")

        mock_claude.assert_called_once()
        assert result == mock_instance

    @patch('src.dependencies.CodexCLI')
    def test_get_codex_provider(self, mock_codex):
        """Test getting Codex provider"""
        from src.dependencies import get_llm_provider

        mock_instance = MagicMock()
        mock_codex.return_value = mock_instance

        result = get_llm_provider("codex")

        mock_codex.assert_called_once()
        assert result == mock_instance

    def test_get_unknown_provider(self):
        """Test getting unknown provider raises ValueError"""
        from src.dependencies import get_llm_provider

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("unknown_provider")


class TestInitDependencies:
    """Test init_dependencies function"""

    @patch('src.dependencies.EmbeddingModel')
    @patch('src.dependencies.QdrantManager')
    def test_init_dependencies_success(self, mock_qdrant_cls, mock_emb_cls):
        """Test successful initialization of dependencies"""
        from src.dependencies import init_dependencies

        mock_emb = MagicMock()
        mock_qdrant = MagicMock()
        mock_emb_cls.return_value = mock_emb
        mock_qdrant_cls.return_value = mock_qdrant

        emb, qdrant = init_dependencies()

        mock_emb_cls.assert_called_once()
        mock_qdrant_cls.assert_called_once()
        assert emb == mock_emb
        assert qdrant == mock_qdrant


class TestGetEmbeddingModel:
    """Test get_embedding_model function"""

    def test_get_embedding_model_not_initialized(self):
        """Test get_embedding_model raises error when not initialized"""
        import src.dependencies as deps
        original = deps.embedding_model
        deps.embedding_model = None

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                deps.get_embedding_model()
        finally:
            deps.embedding_model = original

    def test_get_embedding_model_success(self):
        """Test get_embedding_model returns model when initialized"""
        import src.dependencies as deps
        original = deps.embedding_model
        mock_model = MagicMock()
        deps.embedding_model = mock_model

        try:
            result = deps.get_embedding_model()
            assert result == mock_model
        finally:
            deps.embedding_model = original


class TestGetQdrantManager:
    """Test get_qdrant_manager function"""

    def test_get_qdrant_manager_not_initialized(self):
        """Test get_qdrant_manager raises error when not initialized"""
        import src.dependencies as deps
        original = deps.qdrant_manager
        deps.qdrant_manager = None

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                deps.get_qdrant_manager()
        finally:
            deps.qdrant_manager = original

    def test_get_qdrant_manager_success(self):
        """Test get_qdrant_manager returns manager when initialized"""
        import src.dependencies as deps
        original = deps.qdrant_manager
        mock_manager = MagicMock()
        deps.qdrant_manager = mock_manager

        try:
            result = deps.get_qdrant_manager()
            assert result == mock_manager
        finally:
            deps.qdrant_manager = original
