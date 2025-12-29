"""Configuration module for Fable RAG System"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fables")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# LLM Configuration
LLM_PROVIDERS_STR = os.getenv("LLM_PROVIDERS", "ollama")
LLM_PROVIDERS = [p.strip() for p in LLM_PROVIDERS_STR.split(",") if p.strip()]
LLM_DEFAULT_PROVIDER = os.getenv("LLM_DEFAULT_PROVIDER", LLM_PROVIDERS[0] if LLM_PROVIDERS else "ollama")
OLLAMA_MODELS_STR = os.getenv("OLLAMA_MODELS", "")
OLLAMA_MODELS = [m.strip() for m in OLLAMA_MODELS_STR.split(",") if m.strip()]

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
