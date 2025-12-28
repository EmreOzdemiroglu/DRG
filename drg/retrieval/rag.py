"""RAG (Retrieval-Augmented Generation) retrieval layer."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..embedding import EmbeddingProvider
from ..vector_store import VectorStore, SearchResult
from ..graph import KG

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Retrieval context with chunks and knowledge graph subgraph."""
    
    chunks: List[Dict[str, Any]]
    kg_subgraph: Optional[Dict[str, Any]] = None
    entities: List[str] = None
    relationships: List[Dict[str, Any]] = None


class RAGRetriever:
    """RAG retriever with knowledge graph context integration."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        knowledge_graph: Optional[KG] = None,
        include_kg_context: bool = True,
    ):
        """Initialize RAG retriever.
        
        Args:
            embedding_provider: Embedding provider for query embedding
            vector_store: Vector store for chunk retrieval
            knowledge_graph: Optional knowledge graph for context enrichment
            include_kg_context: Whether to include KG context in retrieval
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.include_kg_context = include_kg_context and knowledge_graph is not None
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_kg_context: Optional[bool] = None,
    ) -> RetrievalContext:
        """Retrieve relevant chunks with optional KG context.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            filters: Optional metadata filters
            include_kg_context: Override default KG context inclusion
        
        Returns:
            RetrievalContext with chunks and optional KG subgraph
        """
        # Embed query
        query_embedding = self.embedding_provider.embed(query)
        
        # Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )
        
        # Convert to chunk format
        chunks = []
        chunk_ids = []
        for result in search_results:
            chunk_data = {
                "chunk_id": result.chunk_id,
                "text": result.metadata.get("chunk_text", ""),
                "score": result.score,
                "metadata": result.metadata,
            }
            chunks.append(chunk_data)
            chunk_ids.append(result.chunk_id)
        
        # Extract KG context if enabled
        kg_subgraph = None
        entities = []
        relationships = []
        
        use_kg_context = include_kg_context if include_kg_context is not None else self.include_kg_context
        
        if use_kg_context and self.knowledge_graph:
            kg_subgraph, entities, relationships = self._extract_kg_context(chunks)
        
        return RetrievalContext(
            chunks=chunks,
            kg_subgraph=kg_subgraph,
            entities=entities,
            relationships=relationships,
        )
    
    def _extract_kg_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> tuple[Optional[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """Extract knowledge graph context from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Tuple of (kg_subgraph, entities, relationships)
        """
        if not self.knowledge_graph:
            return None, [], []
        
        # Get all entity names from KG
        kg_entity_names = set(self.knowledge_graph.nodes.keys())
        
        # Extract entities from chunks by matching KG entity names in chunk text
        found_entities = set()
        chunk_texts = [chunk.get("text", "") for chunk in chunks]
        
        for entity_name in kg_entity_names:
            # Check if entity appears in any chunk text
            for chunk_text in chunk_texts:
                if entity_name.lower() in chunk_text.lower():
                    found_entities.add(entity_name)
                    break
        
        entities = list(found_entities)
        
        # Build subgraph from found entities
        kg_subgraph = {
            "nodes": [],
            "edges": [],
        }
        relationships = []
        
        # Find nodes and edges related to these entities
        for entity in entities:
            # Find node in KG
            if entity in self.knowledge_graph.nodes:
                node_data = self.knowledge_graph.nodes[entity]
                kg_subgraph["nodes"].append({
                    "id": entity,
                    "type": node_data.get("type"),
                    "metadata": node_data,
                })
                
                # Find edges connected to this node
                for edge in self.knowledge_graph.edges:
                    source, relation, target = edge
                    if source == entity or target == entity:
                        # Avoid duplicates
                        edge_key = (source, relation, target)
                        if edge_key not in [(r["source"], r["relation"], r["target"]) for r in relationships]:
                            kg_subgraph["edges"].append({
                                "source": source,
                                "relation": relation,
                                "target": target,
                            })
                            relationships.append({
                                "source": source,
                                "relation": relation,
                                "target": target,
                            })
        
        return kg_subgraph, entities, relationships
    
    def retrieve_with_metadata_filter(
        self,
        query: str,
        k: int = 10,
        entity_filter: Optional[List[str]] = None,
        topic_filter: Optional[List[str]] = None,
    ) -> RetrievalContext:
        """Retrieve with metadata filtering.
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            entity_filter: Optional list of entities to filter by
            topic_filter: Optional list of topics to filter by
        
        Returns:
            RetrievalContext
        """
        filters = {}
        
        if entity_filter:
            # Note: Entity filtering support depends on vector store capabilities
            # ChromaDB, Qdrant, and Pinecone support metadata filtering
            # This is a basic implementation - advanced filtering can be extended
            filters["entities"] = {"$in": entity_filter}
        
        if topic_filter:
            filters["topic"] = {"$in": topic_filter}
        
        return self.retrieve(query=query, k=k, filters=filters if filters else None)


def create_rag_retriever(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    knowledge_graph: Optional[KG] = None,
    include_kg_context: bool = True,
) -> RAGRetriever:
    """Factory function to create RAG retriever.
    
    Args:
        embedding_provider: Embedding provider
        vector_store: Vector store
        knowledge_graph: Optional knowledge graph
        include_kg_context: Whether to include KG context
    
    Returns:
        RAGRetriever instance
    """
    return RAGRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        include_kg_context=include_kg_context,
    )

