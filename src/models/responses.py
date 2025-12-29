"""Response models for Fable RAG System API"""
from pydantic import BaseModel
from typing import List


class FableResult(BaseModel):
    """Single fable result model"""
    id: int
    title: str
    content: str
    moral: str
    score: float
    language: str
    word_count: int


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[FableResult]
    total_results: int


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    collection_name: str
    total_fables: int
    llm_provider: str


class GenerateResponse(BaseModel):
    """Generate response model"""
    query: str
    answer: str
    sources: List[FableResult]
    llm_provider: str
