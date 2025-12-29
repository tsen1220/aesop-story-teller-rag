"""FastAPI application: Fable RAG System API"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

from src.embeddings import EmbeddingModel
from src.qdrant_manager import QdrantManager
from src.llm import Ollama, GeminiCLI, ClaudeCLI, CodexCLI

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="Fable RAG API",
    description="Fable Story Retrieval-Augmented Generation System API",
    version="1.0.0"
)

# Global variables: model and database connection
embedding_model: Optional[EmbeddingModel] = None
qdrant_manager: Optional[QdrantManager] = None

# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fables")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# LLM Configuration
LLM_PROVIDERS_STR = os.getenv("LLM_PROVIDERS", "ollama")
LLM_PROVIDERS = [p.strip() for p in LLM_PROVIDERS_STR.split(",") if p.strip()]
LLM_DEFAULT_PROVIDER = os.getenv("LLM_DEFAULT_PROVIDER", LLM_PROVIDERS[0] if LLM_PROVIDERS else "ollama")
OLLAMA_MODELS_STR = os.getenv("OLLAMA_MODELS", "")
OLLAMA_MODELS = [m.strip() for m in OLLAMA_MODELS_STR.split(",") if m.strip()]

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


# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return (1-20)")
    score_threshold: Optional[float] = Field(None, ge=0, le=1, description="Similarity score threshold (0-1)")


class FableResult(BaseModel):
    id: int
    title: str
    content: str
    moral: str
    score: float
    language: str
    word_count: int


class SearchResponse(BaseModel):
    query: str
    results: List[FableResult]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    message: str
    collection_name: str
    total_fables: int
    llm_provider: str


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query/question")
    limit: int = Field(3, ge=1, le=10, description="Number of fables to use as context")
    provider: Optional[str] = Field(None, description="LLM provider: ollama, claude_code, gemini_cli, codex")
    ollama_model: Optional[str] = Field(None, description="Model name (required for ollama, e.g., llama3.2:latest)")


class GenerateResponse(BaseModel):
    query: str
    answer: str
    sources: List[FableResult]
    llm_provider: str


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize model and connections on application startup"""
    global embedding_model, qdrant_manager, llm_provider

    print("🚀 Initializing Fable RAG System...")

    # Initialize embedding model
    embedding_model = EmbeddingModel()

    # Connect to Qdrant
    qdrant_manager = QdrantManager()

    # Check if collection exists
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


# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Fable RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """List available LLM providers and models"""
    return {
        "providers": LLM_PROVIDERS,
        "default_provider": LLM_DEFAULT_PROVIDER,
        "ollama_models": OLLAMA_MODELS if "ollama" in LLM_PROVIDERS else []
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        info = qdrant_manager.get_collection_info(COLLECTION_NAME)
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

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_fables(request: SearchRequest):
    """
    Search for similar fables

    - **query**: Search query text (e.g., "a story about honesty")
    - **limit**: Number of results to return (1-20, default 5)
    - **score_threshold**: Similarity score threshold (0-1, optional)
    """
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Vectorize query text
        query_vector = embedding_model.encode_single(request.query)

        # Search for similar vectors
        results = qdrant_manager.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        # Format results
        fable_results = [
            FableResult(
                id=result['id'],
                title=result['payload']['title'],
                content=result['payload']['content'],
                moral=result['payload']['moral'],
                score=result['score'],
                language=result['payload']['language'],
                word_count=result['payload']['word_count']
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=fable_results,
            total_results=len(fable_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_answer(request: GenerateRequest):
    """
    Generate answer using RAG (Retrieval-Augmented Generation)

    - **query**: User question (e.g., "What can we learn about honesty?")
    - **limit**: Number of fables to use as context (1-10, default 3)
    - **provider**: LLM provider (ollama, claude_code, gemini_cli, codex)
    - **model**: Model name for ollama (e.g., llama3.2:latest)
    """
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    # Determine provider
    provider_name = request.provider or LLM_DEFAULT_PROVIDER
    if provider_name not in LLM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider_name}' not available. Available: {LLM_PROVIDERS}"
        )

    # Handle model for Ollama
    selected_model = None
    if provider_name == "ollama":
        selected_model = request.ollama_model or (OLLAMA_MODELS[0] if OLLAMA_MODELS else None)
        if selected_model and selected_model not in OLLAMA_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{selected_model}' not available. Available: {OLLAMA_MODELS}"
            )

    # Get or create LLM provider instance
    cache_key = f"{provider_name}:{selected_model}" if provider_name == "ollama" else provider_name
    if cache_key not in llm_providers_cache:
        try:
            llm_providers_cache[cache_key] = get_llm_provider(provider_name, selected_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize {provider_name}: {str(e)}")

    llm = llm_providers_cache[cache_key]

    try:
        # Step 1: Search for relevant fables
        query_vector = embedding_model.encode_single(request.query)
        results = qdrant_manager.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=request.limit
        )

        # Step 2: Build context from fables
        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result['payload']
            context_parts.append(
                f"Fable {i}: {payload['title']}\n"
                f"Content: {payload['content']}\n"
                f"Moral: {payload['moral']}"
            )
        context = "\n\n".join(context_parts)

        # Step 3: Build prompt for LLM
        prompt = f"""Based on the following fables, answer the user's question.

{context}

User's question: {request.query}

Please provide a helpful answer based on the fables above. Reference specific fables when relevant."""

        # Step 4: Generate answer using LLM
        answer = llm.generate(prompt)

        if answer is None:
            raise HTTPException(status_code=500, detail="LLM failed to generate response")

        # Step 5: Format sources
        sources = [
            FableResult(
                id=result['id'],
                title=result['payload']['title'],
                content=result['payload']['content'],
                moral=result['payload']['moral'],
                score=result['score'],
                language=result['payload']['language'],
                word_count=result['payload']['word_count']
            )
            for result in results
        ]

        provider_info = provider_name
        if provider_name == "ollama":
            provider_info = f"ollama ({selected_model})"

        return GenerateResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            llm_provider=provider_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate failed: {str(e)}")


@app.get("/fables/{fable_id}", tags=["Fables"])
async def get_fable_by_id(fable_id: int):
    """Get specific fable by ID"""
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Use Qdrant's retrieve function
        result = qdrant_manager.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[fable_id]
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Fable with ID {fable_id} not found")

        point = result[0]
        return {
            "id": point.id,
            "title": point.payload['title'],
            "content": point.payload['content'],
            "moral": point.payload['moral'],
            "language": point.payload['language'],
            "word_count": point.payload['word_count']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get fable: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
