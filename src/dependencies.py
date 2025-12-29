"""Dependency injection module for Fable RAG System"""
from typing import Optional

from src.embeddings import EmbeddingModel
from src.qdrant_manager import QdrantManager
from src.llm import Ollama, GeminiCLI, ClaudeCLI, CodexCLI

# Global instances
embedding_model: Optional[EmbeddingModel] = None
qdrant_manager: Optional[QdrantManager] = None

# LLM provider instances cache
llm_providers_cache = {}


def get_llm_provider(provider_name: str, ollama_model: Optional[str] = None):
    """Factory function to create LLM provider instance"""
    if provider_name == "ollama":
        return Ollama(model=ollama_model)
    elif provider_name == "gemini_cli":
        return GeminiCLI()
    elif provider_name == "claude_code":
        return ClaudeCLI()
    elif provider_name == "codex":
        return CodexCLI()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def init_dependencies():
    """Initialize all dependencies on startup"""
    global embedding_model, qdrant_manager

    # Initialize embedding model
    embedding_model = EmbeddingModel()

    # Connect to Qdrant
    qdrant_manager = QdrantManager()

    return embedding_model, qdrant_manager


def get_embedding_model() -> EmbeddingModel:
    """Get embedding model instance"""
    if embedding_model is None:
        raise RuntimeError("Embedding model not initialized")
    return embedding_model


def get_qdrant_manager() -> QdrantManager:
    """Get Qdrant manager instance"""
    if qdrant_manager is None:
        raise RuntimeError("Qdrant manager not initialized")
    return qdrant_manager
