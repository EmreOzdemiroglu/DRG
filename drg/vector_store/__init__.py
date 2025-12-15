"""Vector store abstraction layer."""

from .interface import VectorStore, SearchResult
from .chroma import ChromaVectorStore
from .factory import create_vector_store

__all__ = [
    "VectorStore",
    "SearchResult",
    "ChromaVectorStore",
    "create_vector_store",
]

