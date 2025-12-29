"""Models package for Fable RAG System API"""
from .requests import SearchRequest, GenerateRequest
from .responses import FableResult, SearchResponse, HealthResponse, GenerateResponse

__all__ = [
    "SearchRequest",
    "GenerateRequest",
    "FableResult",
    "SearchResponse",
    "HealthResponse",
    "GenerateResponse",
]
