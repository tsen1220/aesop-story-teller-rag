"""Health and info handlers for Fable RAG System API"""
from fastapi import APIRouter, HTTPException

from src.config import COLLECTION_NAME, LLM_PROVIDERS, LLM_DEFAULT_PROVIDER, OLLAMA_MODELS
import src.dependencies as deps
from src.models import HealthResponse

router = APIRouter()


@router.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Fable RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/models", tags=["Models"])
async def list_models():
    """List available LLM providers and models"""
    return {
        "providers": LLM_PROVIDERS,
        "default_provider": LLM_DEFAULT_PROVIDER,
        "ollama_models": OLLAMA_MODELS if "ollama" in LLM_PROVIDERS else []
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if deps.embedding_model is None or deps.qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        info = deps.qdrant_manager.get_collection_info(COLLECTION_NAME)
        if info is None:
            raise HTTPException(
                status_code=503,
                detail=f"Collection '{COLLECTION_NAME}' does not exist, please run init_database.py first"
            )

        return HealthResponse(
            status="healthy",
            message="System running normally",
            collection_name=COLLECTION_NAME,
            total_fables=info['points_count'],
            llm_provider=f"{', '.join(LLM_PROVIDERS)} (default: {LLM_DEFAULT_PROVIDER})"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
