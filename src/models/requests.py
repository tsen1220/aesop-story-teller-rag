"""Request models for Fable RAG System API"""
from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return (1-20)")
    score_threshold: Optional[float] = Field(None, ge=0, le=1, description="Similarity score threshold (0-1)")


class GenerateRequest(BaseModel):
    """Generate request model"""
    query: str = Field(..., min_length=1, description="User query/question")
    limit: int = Field(3, ge=1, le=10, description="Number of fables to use as context")
    provider: Optional[str] = Field(None, description="LLM provider: ollama, claude_code, gemini_cli, codex")
    ollama_model: Optional[str] = Field(None, description="Model name (required for ollama, e.g., llama3.2:latest)")
