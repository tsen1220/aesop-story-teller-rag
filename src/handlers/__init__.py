"""Handlers package for Fable RAG System API"""
from fastapi import APIRouter

from .health import router as health_router
from .search import router as search_router
from .generate import router as generate_router
from .fables import router as fables_router

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(health_router)
router.include_router(search_router)
router.include_router(generate_router)
router.include_router(fables_router)

__all__ = ["router"]
