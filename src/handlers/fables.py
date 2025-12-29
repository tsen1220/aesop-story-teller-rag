"""Fables handler for Fable RAG System API"""
from fastapi import APIRouter, HTTPException

from src.config import COLLECTION_NAME
import src.dependencies as deps

router = APIRouter()


@router.get("/fables/{fable_id}", tags=["Fables"])
async def get_fable_by_id(fable_id: int):
    """Get specific fable by ID"""
    if deps.qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Use Qdrant's retrieve function
        result = deps.qdrant_manager.client.retrieve(
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
