"""Chroma vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import uuid

from .interface import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Chroma vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "drg_chunks",
        persist_directory: Optional[str] = None,
        embedding_function=None,
    ):
        """Initialize Chroma vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Optional directory to persist data
            embedding_function: Optional embedding function (for Chroma's auto-embedding)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to Chroma store."""
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have same length")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        if len(ids) != len(embeddings):
            raise ValueError("ids must have same length as embeddings")
        
        # Extract chunk_id from metadata if available
        chunk_ids = [m.get("chunk_id", id) for m, id in zip(metadata, ids)]
        
        # Add to Chroma
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadata,
            ids=chunk_ids,
        )
        
        logger.info(f"Added {len(embeddings)} items to collection")
        return chunk_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        # Convert filters to Chroma format
        where = None
        if filters:
            where = {}
            for key, value in filters.items():
                where[key] = value
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )
        
        # Convert to SearchResult format
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                # Convert distance to similarity score (Chroma uses distance, we want similarity)
                score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                
                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        score=score,
                        metadata=metadata,
                    )
                )
        
        return search_results
    
    def update(
        self,
        id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an item in Chroma store."""
        # Chroma doesn't have direct update, so we delete and re-add
        # Get current data first
        current = self.collection.get(ids=[id])
        
        if not current['ids']:
            raise ValueError(f"Item {id} not found")
        
        # Prepare new data
        new_embedding = embedding if embedding else current['embeddings'][0]
        new_metadata = metadata if metadata else current['metadatas'][0]
        
        # Delete and re-add
        self.collection.delete(ids=[id])
        self.collection.add(
            embeddings=[new_embedding],
            metadatas=[new_metadata],
            ids=[id],
        )
    
    def delete(self, ids: List[str]):
        """Delete items from Chroma store."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} items from collection")
    
    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item."""
        results = self.collection.get(ids=[id])
        if results['ids']:
            return results['metadatas'][0] if results['metadatas'] else {}
        return None
    
    def count(self) -> int:
        """Get total number of items in store."""
        return self.collection.count()

