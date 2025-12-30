"""Vector store factory functions."""

from typing import Optional
from .interface import VectorStore
from .chroma import ChromaVectorStore


def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "drg_chunks",
    persist_directory: Optional[str] = None,
    **kwargs
) -> VectorStore:
    """Factory function to create vector store.
    
    Args:
        store_type: Store type ("chroma", "qdrant", "pinecone", "faiss")
        collection_name: Collection/database name
        persist_directory: Optional directory to persist data
        **kwargs: Additional store-specific parameters
    
    Returns:
        VectorStore instance
    
    Examples:
        >>> # ChromaDB (default)
        >>> store = create_vector_store("chroma", collection_name="my_chunks")
        
        >>> # Qdrant
        >>> store = create_vector_store("qdrant", collection_name="my_chunks", url="http://localhost:6333")
        
        >>> # Pinecone
        >>> store = create_vector_store("pinecone", index_name="my-index", api_key="...", dimension=1536)
        
        >>> # FAISS
        >>> store = create_vector_store("faiss", collection_name="my_chunks", dimension=1536, persist_directory="./faiss_index")
    """
    store_type_lower = store_type.lower()
    
    if store_type_lower == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
    
    elif store_type_lower == "qdrant":
        try:
            from .qdrant import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "Qdrant vector store requires qdrant-client. "
                "Install with: pip install qdrant-client or pip install drg[qdrant]"
            )
        return QdrantVectorStore(
            collection_name=collection_name,
            **kwargs
        )
    
    elif store_type_lower == "pinecone":
        try:
            from .pinecone import PineconeVectorStore
        except ImportError:
            raise ImportError(
                "Pinecone vector store requires pinecone-client. "
                "Install with: pip install pinecone-client or pip install drg[pinecone]"
            )
        return PineconeVectorStore(
            collection_name=collection_name,
            **kwargs
        )
    
    elif store_type_lower == "faiss":
        try:
            from .faiss import FAISSVectorStore
        except ImportError:
            raise ImportError(
                "FAISS vector store requires faiss-cpu or faiss-gpu. "
                "Install with: pip install faiss-cpu or pip install drg[faiss]"
            )
        return FAISSVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown vector store type: {store_type}. Supported types: chroma, qdrant, pinecone, faiss")

