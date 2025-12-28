"""Vector store abstraction layer."""

from .interface import VectorStore, SearchResult
from .chroma import ChromaVectorStore
from .factory import create_vector_store

# Optional vector stores (lazy imports to avoid requiring dependencies)
__all__ = [
    "VectorStore",
    "SearchResult",
    "ChromaVectorStore",
    "create_vector_store",
]

# Optional exports (available if dependencies are installed)
try:
    from .qdrant import QdrantVectorStore
    __all__.append("QdrantVectorStore")
except ImportError:
    pass

try:
    from .pinecone import PineconeVectorStore
    __all__.append("PineconeVectorStore")
except ImportError:
    pass

try:
    from .faiss import FAISSVectorStore
    __all__.append("FAISSVectorStore")
except ImportError:
    pass

