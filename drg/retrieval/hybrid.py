"""Hybrid retriever combining RAG and GraphRAG."""

import logging
from typing import List, Dict, Any, Optional

from .rag import RAGRetriever, RetrievalContext
from .drg_search import DRGSearch
from .graphrag import GraphRAGRetriever, GraphRAGRetrievalContext

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining RAG and GraphRAG.
    
    Supports both classic RAG (vector search) and GraphRAG (KG traversal + community reports).
    """
    
    def __init__(
        self,
        rag_retriever: Optional[RAGRetriever] = None,
        graphrag_retriever: Optional[GraphRAGRetriever] = None,
        drg_search: Optional[DRGSearch] = None,
        fusion_method: str = "reciprocal_rank",
        use_graphrag: bool = True,
    ):
        """Initialize hybrid retriever.
        
        Args:
            rag_retriever: Optional RAG retriever for vector search (classic RAG)
            graphrag_retriever: Optional GraphRAG retriever (GraphRAG approach)
            drg_search: Optional DRG search for graph traversal (legacy)
            fusion_method: Fusion method ("reciprocal_rank" or "weighted")
            use_graphrag: Whether to use GraphRAG approach (default: True)
        """
        self.rag_retriever = rag_retriever
        self.graphrag_retriever = graphrag_retriever
        self.drg_search = drg_search
        self.fusion_method = fusion_method
        self.use_graphrag = use_graphrag
        
        # Validate configuration
        if use_graphrag and graphrag_retriever is None:
            raise ValueError("GraphRAG retriever required when use_graphrag=True")
        if not use_graphrag and rag_retriever is None:
            raise ValueError("RAG retriever required when use_graphrag=False")
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        rag_k: int = 10,
        drg_k: int = 10,
    ) -> RetrievalContext:
        """Retrieve using both RAG and GraphRAG/DRG, then fuse results.
        
        Args:
            query: Query text
            k: Final number of results
            rag_k: Number of RAG results (if using classic RAG)
            drg_k: Number of DRG/GraphRAG results
        
        Returns:
            RetrievalContext with fused results
        """
        if self.use_graphrag and self.graphrag_retriever:
            # GraphRAG approach: KG traversal + Community reports
            graphrag_context = self.graphrag_retriever.retrieve(query=query, k_entities=drg_k)
            
            # Convert GraphRAG context to RetrievalContext format
            # Note: GraphRAG doesn't return chunks, only KG subgraph + reports
            return RetrievalContext(
                chunks=[],  # GraphRAG doesn't use chunks
                kg_subgraph=graphrag_context.kg_subgraph,
                entities=graphrag_context.entities,
                relationships=graphrag_context.relationships,
            )
        else:
            # Classic RAG + DRG hybrid
            if self.rag_retriever is None:
                raise ValueError("RAG retriever required for classic RAG mode")
            
            # RAG retrieval
            rag_context = self.rag_retriever.retrieve(query=query, k=rag_k)
            
            # DRG search (if available)
            if self.drg_search:
                drg_results = self.drg_search.weighted_search(query=query, k=drg_k)
                
                # Fuse results
                if self.fusion_method == "reciprocal_rank":
                    fused_chunks = self._reciprocal_rank_fusion(
                        rag_chunks=rag_context.chunks,
                        drg_results=drg_results,
                        k=k,
                    )
                else:
                    # Default: just combine
                    fused_chunks = rag_context.chunks[:k]
            else:
                fused_chunks = rag_context.chunks[:k]
            
            return RetrievalContext(
                chunks=fused_chunks,
                kg_subgraph=rag_context.kg_subgraph,
                entities=rag_context.entities,
                relationships=rag_context.relationships,
            )
    
    def _reciprocal_rank_fusion(
        self,
        rag_chunks: List[Dict[str, Any]],
        drg_results: List[Dict[str, Any]],
        k: int = 10,
        k_param: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion of RAG and DRG results.
        
        Args:
            rag_chunks: RAG retrieval results
            drg_results: DRG search results
            k: Number of final results
            k_param: RRF parameter (typically 60)
        
        Returns:
            Fused list of chunks
        """
        # Create score map
        scores = {}
        
        # Add RAG scores
        for rank, chunk in enumerate(rag_chunks, start=1):
            chunk_id = chunk.get("chunk_id", str(rank))
            rag_score = 1.0 / (k_param + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rag_score
        
        # Add DRG scores (convert to chunk format)
        for rank, node in enumerate(drg_results, start=1):
            # Try to find chunks related to this entity
            entity = node.get("entity", "")
            # Simplified: use entity as chunk_id
            drg_score = 1.0 / (k_param + rank)
            scores[entity] = scores.get(entity, 0.0) + drg_score
        
        # Sort by combined score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct chunks (simplified - in production, map back to actual chunks)
        fused_chunks = []
        for chunk_id, score in sorted_items[:k]:
            # Find original chunk or create from DRG result
            chunk = next((c for c in rag_chunks if c.get("chunk_id") == chunk_id), None)
            if chunk:
                chunk["combined_score"] = score
                fused_chunks.append(chunk)
        
        return fused_chunks

