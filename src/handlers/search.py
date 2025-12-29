"""Search handler for Fable RAG System API"""
from fastapi import APIRouter, HTTPException

from src.config import COLLECTION_NAME
import src.dependencies as deps
from src.models import SearchRequest, SearchResponse, FableResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_fables(request: SearchRequest):
    """
    Search for similar fables

    - **query**: Search query text (e.g., "a story about honesty")
    - **limit**: Number of results to return (1-20, default 5)
    - **score_threshold**: Similarity score threshold (0-1, optional)
    """
    if deps.embedding_model is None or deps.qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Vectorize query text
        query_vector = deps.embedding_model.encode_single(request.query)

        # Search for similar vectors
        results = deps.qdrant_manager.search(
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
