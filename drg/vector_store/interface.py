"""Vector store interface definitions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result with metadata."""
    
    chunk_id: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorStore(ABC):
    """Abstract vector store interface."""
    
    @abstractmethod
    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to store.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: Optional list of IDs (auto-generated if not provided)
        
        Returns:
            List of IDs for added items
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def update(
        self,
        id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an item in the store.
        
        Args:
            id: Item ID
            embedding: Optional new embedding
            metadata: Optional new metadata
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete items from store.
        
        Args:
            ids: List of IDs to delete
        """
        pass
    
    @abstractmethod
    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item.
        
        Args:
            id: Item ID
        
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of items in store.
        
        Returns:
            Total count
        """
        pass

