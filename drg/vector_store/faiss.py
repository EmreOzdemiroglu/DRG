"""FAISS vector store implementation."""

import logging
from typing import List, Dict, Any, Optional
import uuid
import pickle
from pathlib import Path

from .interface import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "drg_chunks",
        dimension: int = 1536,
        index_type: str = "flat",
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """Initialize FAISS vector store.
        
        Args:
            collection_name: Name of the collection/index
            dimension: Vector dimension (required)
            index_type: FAISS index type ("flat", "ivf", "hnsw") - "flat" is simplest and most accurate
            persist_directory: Optional directory to persist index to disk
            **kwargs: Additional FAISS index parameters
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu or faiss-gpu is required. Install with: pip install faiss-cpu or pip install drg[faiss]"
            )
        
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.persist_directory = persist_directory
        self.np = np
        
        # Create FAISS index based on type
        if self.index_type == "flat":
            # Flat index (brute force, most accurate)
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
            # Convert to cosine similarity by normalizing vectors
            self.use_cosine = kwargs.get("use_cosine", True)
        elif self.index_type == "ivf":
            # IVF (Inverted File Index) - faster for large datasets
            nlist = kwargs.get("nlist", 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.use_cosine = kwargs.get("use_cosine", True)
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) - approximate, very fast
            m = kwargs.get("m", 32)  # Number of connections
            self.index = faiss.IndexHNSWFlat(dimension, m)
            self.use_cosine = kwargs.get("use_cosine", True)
        else:
            raise ValueError(f"Unknown index type: {index_type}. Use 'flat', 'ivf', or 'hnsw'")
        
        # Storage for metadata (FAISS only stores vectors, not metadata)
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}  # Map ID to FAISS index position
        self.index_to_id: Dict[int, str] = {}  # Map FAISS index position to ID
        
        # Load from disk if persist_directory exists
        if persist_directory:
            self._load_from_disk()
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            # IVF needs training, but we'll train on first batch of vectors
            logger.warning("IVF index requires training. Will train on first batch of vectors.")
    
    def _normalize_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """Normalize vectors for cosine similarity."""
        if not self.use_cosine:
            return vectors
        
        vectors_np = self.np.array(vectors, dtype=self.np.float32)
        norms = self.np.linalg.norm(vectors_np, axis=1, keepdims=True)
        norms = self.np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = vectors_np / norms
        return normalized.tolist()
    
    def _save_to_disk(self):
        """Save index and metadata to disk."""
        if not self.persist_directory:
            return
        
        import faiss
        
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = persist_path / f"{self.collection_name}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = persist_path / f"{self.collection_name}_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
            }, f)
    
    def _load_from_disk(self):
        """Load index and metadata from disk."""
        if not self.persist_directory:
            return
        
        persist_path = Path(self.persist_directory)
        index_path = persist_path / f"{self.collection_name}.index"
        metadata_path = persist_path / f"{self.collection_name}_metadata.pkl"
        
        if index_path.exists():
            import faiss
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded existing index: {index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                # Will create a new index below
        
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata_store = data.get("metadata_store", {})
                self.id_to_index = data.get("id_to_index", {})
                self.index_to_id = data.get("index_to_id", {})
            logger.info(f"Loaded metadata: {len(self.metadata_store)} items")
    
    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to FAISS store."""
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have same length")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        if len(ids) != len(embeddings):
            raise ValueError("ids must have same length as embeddings")
        
        # Validate vector dimensions
        for emb in embeddings:
            if len(emb) != self.dimension:
                raise ValueError(f"All embeddings must have dimension {self.dimension}")
        
        # Normalize vectors for cosine similarity if needed
        normalized_embeddings = self._normalize_vectors(embeddings)
        
        # Convert to numpy array
        vectors_np = self.np.array(normalized_embeddings, dtype=self.np.float32)
        
        # Train IVF index if needed and not yet trained
        if self.index_type == "ivf" and not self.index.is_trained:
            if len(vectors_np) >= self.index.nlist:  # Need at least nlist vectors to train
                self.index.train(vectors_np)
                logger.info("Trained IVF index")
            else:
                logger.warning(f"Need at least {self.index.nlist} vectors to train IVF index")
        
        # Add to FAISS index
        start_index = self.index.ntotal
        self.index.add(vectors_np)
        
        # Store metadata and ID mappings
        for i, (id_val, meta) in enumerate(zip(ids, metadata)):
            index_pos = start_index + i
            self.metadata_store[id_val] = meta
            self.id_to_index[id_val] = index_pos
            self.index_to_id[index_pos] = id_val
        
        # Save to disk if persist_directory is set
        if self.persist_directory:
            self._save_to_disk()
        
        logger.info(f"Added {len(embeddings)} items to index")
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding must have dimension {self.dimension}")
        
        # Normalize query vector
        normalized_query = self._normalize_vectors([query_embedding])[0]
        query_np = self.np.array([normalized_query], dtype=self.np.float32)
        
        # Search
        distances, indices = self.index.search(query_np, min(k, self.index.ntotal))
        
        # Convert to SearchResult format
        search_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            
            id_val = self.index_to_id.get(idx)
            if id_val is None:
                continue
            
            # Apply filters if provided
            if filters:
                metadata = self.metadata_store.get(id_val, {})
                if not all(metadata.get(key) == value for key, value in filters.items()):
                    continue
            
            # Convert distance to similarity score
            # For cosine similarity (normalized vectors), similarity = 1 - distance/2
            # For L2 distance, we use 1 / (1 + distance)
            if self.use_cosine:
                score = 1.0 - (distance / 2.0)  # Cosine distance to similarity
            else:
                score = 1.0 / (1.0 + distance)  # L2 distance to similarity
            
            search_results.append(
                SearchResult(
                    chunk_id=id_val,
                    score=float(score),
                    metadata=self.metadata_store.get(id_val, {}),
                )
            )
        
        return search_results
    
    def update(
        self,
        id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an item in FAISS store.
        
        Note: FAISS doesn't support in-place updates efficiently.
        This implementation deletes and re-adds the vector.
        """
        if id not in self.id_to_index:
            raise ValueError(f"Item {id} not found")
        
        # Get current data
        current_metadata = self.metadata_store.get(id, {})
        current_index = self.id_to_index[id]
        
        # Prepare new data
        new_metadata = metadata if metadata else current_metadata
        
        if embedding is not None:
            # Need to replace the vector - FAISS doesn't support updates well
            # We'll delete and re-add
            self.delete([id])
            self.add([embedding], [new_metadata], [id])
        else:
            # Just update metadata
            self.metadata_store[id] = new_metadata
            if self.persist_directory:
                self._save_to_disk()
    
    def delete(self, ids: List[str]):
        """Delete items from FAISS store.
        
        Note: FAISS doesn't support efficient deletion.
        This is a limitation of FAISS - for production use, consider rebuilding the index.
        """
        # FAISS doesn't support deletion efficiently
        # Mark as deleted in metadata store and rebuild mappings
        for id_val in ids:
            if id_val in self.id_to_index:
                index_pos = self.id_to_index[id_val]
                del self.id_to_index[id_val]
                del self.index_to_id[index_pos]
                if id_val in self.metadata_store:
                    del self.metadata_store[id_val]
        
        logger.warning(
            "FAISS doesn't support efficient deletion. "
            "Deleted items are removed from metadata but vectors remain in index. "
            "Consider rebuilding the index for production use."
        )
        
        if self.persist_directory:
            self._save_to_disk()
    
    def get_metadata(self, id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item."""
        return self.metadata_store.get(id)
    
    def count(self) -> int:
        """Get total number of items in store."""
        return len(self.metadata_store)

