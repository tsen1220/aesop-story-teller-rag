"""FastAPI application: Fable RAG System API - Entrypoint"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.config import COLLECTION_NAME, LLM_PROVIDERS, LLM_DEFAULT_PROVIDER, OLLAMA_MODELS, API_HOST, API_PORT
from src.dependencies import init_dependencies, qdrant_manager
from src.handlers import router

# Create FastAPI application
app = FastAPI(
    title="Fable RAG API",
    description="Fable Story Retrieval-Augmented Generation System API",
    version="1.0.0"
)

# Setup CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize model and connections on application startup"""
    print("🚀 Initializing Fable RAG System...")

    # Initialize dependencies
    init_dependencies()

    # Check if collection exists
    from src.dependencies import qdrant_manager
    info = qdrant_manager.get_collection_info(COLLECTION_NAME)
    if info:
        print(f"✓ Collection '{COLLECTION_NAME}' ready with {info['points_count']} fables")
    else:
        print(f"⚠ Collection '{COLLECTION_NAME}' does not exist, please run init_database.py first")

    # Initialize LLM providers
    print(f"✓ Available LLM providers: {', '.join(LLM_PROVIDERS)}")
    print(f"  Default provider: {LLM_DEFAULT_PROVIDER}")
    if "ollama" in LLM_PROVIDERS and OLLAMA_MODELS:
        print(f"  Ollama models: {', '.join(OLLAMA_MODELS)}")

    print("✓ System startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("👋 Shutting down system...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
