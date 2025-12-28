"""Pinecone vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import uuid

from .interface import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "drg-chunks",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **kwargs
    ):
        """Initialize Pinecone vector store.
        
        Args:
            collection_name: Name of the index (legacy parameter name, use index_name)
            api_key: Pinecone API key (required)
            environment: Pinecone environment/region (e.g., "us-east-1-aws")
            index_name: Name of the Pinecone index (defaults to collection_name)
            dimension: Vector dimension (required for index creation)
            metric: Distance metric ("cosine", "euclidean", "dotproduct")
            **kwargs: Additional Pinecone client parameters
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "pinecone-client is required. Install with: pip install pinecone-client or pip install drg[pinecone]"
            )
        
        self.index_name = index_name or collection_name
        
        # Initialize Pinecone
        if api_key:
            pinecone.init(api_key=api_key, environment=environment, **kwargs)
        else:
            # Try to get from environment
            import os
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key is required. Provide via api_key parameter or PINECONE_API_KEY env var")
            environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
            pinecone.init(api_key=api_key, environment=environment, **kwargs)
        
        # Get or create index
        try:
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to existing index: {self.index_name}")
        except Exception as e:
            if dimension is None:
                raise ValueError(
                    f"Index {self.index_name} does not exist. "
                    "dimension parameter is required to create a new index."
                ) from e
            
            # Create index
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
            )
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Created new index: {self.index_name} with dimension {dimension}")
    
    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to Pinecone store."""
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have same length")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        if len(ids) != len(embeddings):
            raise ValueError("ids must have same length as embeddings")
        
        # Prepare vectors for Pinecone (format: [(id, vector, metadata), ...])
        vectors_to_upsert = [
            (id_val, emb, meta)
            for id_val, emb, meta in zip(ids, embeddings, metadata)
        ]
        
        # Add to Pinecone (upsert in batches)
        batch_size = 100  # Pinecone recommends batches of up to 100 vectors
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(embeddings)} items to index")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        # Convert filters to Pinecone filter format (metadata filter)
        metadata_filter = filters
        
        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=metadata_filter,
        )
        
        # Convert to SearchResult format
        search_results = [
            SearchResult(
                chunk_id=str(match.id),
                score=match.score,
                metadata=match.metadata or {},
            )
            for match in results.matches
        ]
        
        return search_results
    
    def update(
        self,
        id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an item in Pinecone store."""
        # Pinecone uses upsert for updates
        # Get current vector if embedding not provided
        if embedding is None or metadata is None:
            # Fetch current data
            fetch_result = self.index.fetch(ids=[id])
            if id not in fetch_result.vectors:
                raise ValueError(f"Item {id} not found")
            
            current_vector = fetch_result.vectors[id].values
            current_metadata = fetch_result.vectors[id].metadata or {}
            
            new_vector = embedding if embedding else current_vector
            new_metadata = metadata if metadata else current_metadata
        else:
            new_vector = embedding
            new_metadata = metadata
        
        # Upsert (update or insert)
        self.index.upsert(vectors=[(id, new_vector, new_metadata)])
    
    def delete(self, ids: List[str]):
        """Delete items from Pinecone store."""
        self.index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} items from index")
    
    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item."""
        fetch_result = self.index.fetch(ids=[id])
        if id in fetch_result.vectors:
            return fetch_result.vectors[id].metadata or {}
        return None
    
    def count(self) -> int:
        """Get total number of items in store."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count

