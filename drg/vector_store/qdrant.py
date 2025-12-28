"""Qdrant vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import uuid

from .interface import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "drg_chunks",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        **kwargs
    ):
        """Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            url: Qdrant server URL (default: "http://localhost:6333")
            api_key: Optional API key for Qdrant Cloud
            prefer_grpc: Whether to prefer gRPC over REST API
            **kwargs: Additional Qdrant client parameters
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client or pip install drg[qdrant]"
            )
        
        self.collection_name = collection_name
        self.url = url or kwargs.get("host", "http://localhost:6333")
        
        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                **{k: v for k, v in kwargs.items() if k != "host"}
            )
        else:
            self.client = QdrantClient(
                url=self.url,
                prefer_grpc=prefer_grpc,
                **{k: v for k, v in kwargs.items() if k != "host"}
            )
        
        # Get or create collection
        try:
            collection_info = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
            self.vector_size = collection_info.config.params.vectors.size
        except Exception:
            # Collection doesn't exist, need vector_size to create
            vector_size = kwargs.get("vector_size", 1536)  # Default OpenAI embedding size
            self.vector_size = vector_size
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created new collection: {collection_name} with vector size {vector_size}")
    
    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to Qdrant store."""
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have same length")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        if len(ids) != len(embeddings):
            raise ValueError("ids must have same length as embeddings")
        
        # Validate vector sizes
        vector_size = len(embeddings[0]) if embeddings else self.vector_size
        for emb in embeddings:
            if len(emb) != vector_size:
                raise ValueError(f"All embeddings must have same size: {vector_size}")
        
        # Prepare points for Qdrant
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=id_val,
                vector=emb,
                payload=meta,
            )
            for id_val, emb, meta in zip(ids, embeddings, metadata)
        ]
        
        # Add to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info(f"Added {len(embeddings)} items to collection")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Convert filters to Qdrant filter format
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
        )
        
        # Convert to SearchResult format
        search_results = [
            SearchResult(
                chunk_id=str(hit.id),
                score=hit.score,
                metadata=hit.payload or {},
            )
            for hit in results
        ]
        
        return search_results
    
    def update(
        self,
        id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an item in Qdrant store."""
        from qdrant_client.models import PointStruct
        
        # Get current point
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
        )
        
        if not points:
            raise ValueError(f"Item {id} not found")
        
        current_point = points[0]
        
        # Prepare new data
        new_vector = embedding if embedding else current_point.vector
        new_payload = metadata if metadata else current_point.payload or {}
        
        # Update (upsert)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=new_vector, payload=new_payload)],
        )
    
    def delete(self, ids: List[str]):
        """Delete items from Qdrant store."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        logger.info(f"Deleted {len(ids)} items from collection")
    
    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item."""
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
        )
        if points:
            return points[0].payload or {}
        return None
    
    def count(self) -> int:
        """Get total number of items in store."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

