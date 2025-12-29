"""Generate handler for Fable RAG System API"""
from fastapi import APIRouter, HTTPException

from src.config import COLLECTION_NAME, LLM_PROVIDERS, LLM_DEFAULT_PROVIDER, OLLAMA_MODELS
import src.dependencies as deps
from src.models import GenerateRequest, GenerateResponse, FableResult

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, tags=["Generate"])
async def generate_answer(request: GenerateRequest):
    """
    Generate answer using RAG (Retrieval-Augmented Generation)

    - **query**: User question (e.g., "What can we learn about honesty?")
    - **limit**: Number of fables to use as context (1-10, default 3)
    - **provider**: LLM provider (ollama, claude_code, gemini_cli, codex)
    - **ollama_model**: Model name for ollama (e.g., llama3.2:latest)
    """
    if deps.embedding_model is None or deps.qdrant_manager is None:
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
    if cache_key not in deps.llm_providers_cache:
        try:
            deps.llm_providers_cache[cache_key] = deps.get_llm_provider(provider_name, selected_model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize {provider_name}: {str(e)}")

    llm = deps.llm_providers_cache[cache_key]

    try:
        # Step 1: Search for relevant fables
        query_vector = deps.embedding_model.encode_single(request.query)
        results = deps.qdrant_manager.search(
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
