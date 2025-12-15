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
    """
    store_type_lower = store_type.lower()
    
    if store_type_lower == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )
    
    elif store_type_lower == "qdrant":
        # TODO: Implement QdrantVectorStore
        raise NotImplementedError("Qdrant vector store not yet implemented")
    
    elif store_type_lower == "pinecone":
        # TODO: Implement PineconeVectorStore
        raise NotImplementedError("Pinecone vector store not yet implemented")
    
    elif store_type_lower == "faiss":
        # TODO: Implement FAISSVectorStore
        raise NotImplementedError("FAISS vector store not yet implemented")
    
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

