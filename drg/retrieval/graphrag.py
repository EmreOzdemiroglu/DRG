"""GraphRAG retrieval - GraphRAG yapısına uygun retrieval.

GraphRAG'de:
- Query'de embed sadece seed entity bulmak için kullanılır
- Ana bilgi kaynağı: KG traversal + Community reportlar
- Vector store query'de kullanılır (sadece bağlam farkına varmak için, çok yüksek veri olduğunda)
- Asıl bilgi yine KG'den gelir
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import deque

from ..embedding import EmbeddingProvider
from ..vector_store import VectorStore
from ..graph import KG
from ..graph.kg_core import EnhancedKG, KGNode
from ..graph.community_report import CommunityReport, CommunityReportGenerator
from ..graph.visualization_adapter import ProvenanceGraph, ProvenanceNode, ProvenanceEdge

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGRetrievalContext:
    """GraphRAG retrieval context - KG subgraph + Community reports."""
    
    kg_subgraph: Dict[str, Any]
    entities: List[str]
    relationships: List[Dict[str, Any]]
    community_reports: List[Dict[str, Any]]
    seed_entities: List[str]
    context_chunks: Optional[List[Dict[str, Any]]] = None  # Vector store'dan gelen bağlam (opsiyonel)
    provenance_graph: Optional[Dict[str, Any]] = None  # Provenance graph for explainability


class GraphRAGRetriever:
    """GraphRAG retriever - GraphRAG yapısına uygun retrieval.
    
    Query'de:
    1. Query'yi embed et
    2. (Opsiyonel) Vector store'da query yap → Bağlam farkına varmak için (çok yüksek veri olduğunda)
    3. KG'deki entity embedding'leriyle match et → Seed entities
    4. Seed entities'den graph traversal yap → Subgraph
    5. Seed entities'in bulunduğu cluster'ları bul → Community reports
    6. Final context = Subgraph + Community reports (Ana bilgi kaynağı)
    
    Not: Vector store query sadece "bağlam farkına varmak" için kullanılır.
         Asıl bilgi kaynağı KG traversal + Community reports'tur.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        knowledge_graph: EnhancedKG,
        community_reports: Optional[List[CommunityReport]] = None,
        vector_store: Optional[VectorStore] = None,
        use_vector_store_context: bool = False,
        similarity_threshold: float = 0.5,
        max_hops: int = 2,
    ):
        """Initialize GraphRAG retriever.
        
        Args:
            embedding_provider: Embedding provider for query embedding (seed entity matching)
            knowledge_graph: Enhanced knowledge graph with entity embeddings
            community_reports: Optional pre-generated community reports
            vector_store: Optional vector store for context awareness (çok yüksek veri için)
            use_vector_store_context: Whether to use vector store for context awareness
            similarity_threshold: Minimum similarity for seed entity matching
            max_hops: Maximum graph traversal hops
        """
        self.embedding_provider = embedding_provider
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.use_vector_store_context = use_vector_store_context and vector_store is not None
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
        
        # Generate community reports if not provided
        if community_reports is None:
            logger.info("Generating community reports...")
            report_generator = CommunityReportGenerator(self.kg)
            self.community_reports = {
                report.cluster_id: report 
                for report in report_generator.generate_all_reports()
            }
        else:
            self.community_reports = {
                report.cluster_id: report 
                for report in community_reports
            }
        
        logger.info(f"Initialized GraphRAG retriever with {len(self.community_reports)} community reports")
    
    def retrieve(
        self,
        query: str,
        k_entities: int = 10,
        k_reports: int = 5,
        k_context_chunks: int = 5,
    ) -> GraphRAGRetrievalContext:
        """Retrieve using GraphRAG approach.
        
        Args:
            query: Query text
            k_entities: Maximum number of seed entities to find
            k_reports: Maximum number of community reports to return
            k_context_chunks: Maximum number of context chunks from vector store (bağlam için)
        
        Returns:
            GraphRAGRetrievalContext with subgraph and community reports
        """
        # Step 1: Embed query
        query_embedding = self.embedding_provider.embed(query)
        
        # Step 2: (Opsiyonel) Vector store'da query yap - Bağlam farkına varmak için
        context_chunks = None
        if self.use_vector_store_context:
            logger.info("Using vector store for context awareness...")
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k_context_chunks,
            )
            context_chunks = [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.metadata.get("chunk_text", ""),
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in search_results
            ]
            logger.info(f"Retrieved {len(context_chunks)} context chunks from vector store")
        
        # Step 3: Find seed entities by matching entity embeddings
        seed_entities = self._find_seed_entities(
            query_embedding=query_embedding,
            k=k_entities,
        )
        
        if not seed_entities:
            logger.warning("No seed entities found for query")
            return GraphRAGRetrievalContext(
                kg_subgraph={"nodes": [], "edges": []},
                entities=[],
                relationships=[],
                community_reports=[],
                seed_entities=[],
                context_chunks=context_chunks,
                provenance_graph=None,
            )
        
        logger.info(f"Found {len(seed_entities)} seed entities: {seed_entities[:5]}")
        
        # Step 4: Graph traversal from seed entities (Ana bilgi kaynağı)
        subgraph, entities, relationships = self._traverse_graph(
            seed_entities=seed_entities,
            max_hops=self.max_hops,
        )
        
        # Step 5: Find relevant community reports (Ana bilgi kaynağı)
        relevant_reports = self._find_relevant_reports(
            seed_entities=seed_entities,
            k=k_reports,
        )
        
        # Generate provenance graph for explainability
        provenance_graph = self._create_provenance_graph(
            query=query,
            seed_entities=seed_entities,
            context_chunks=context_chunks,
            relevant_reports=relevant_reports,
            entities=entities,
            relationships=relationships,
        )
        
        return GraphRAGRetrievalContext(
            kg_subgraph=subgraph,
            entities=entities,
            relationships=relationships,
            community_reports=relevant_reports,
            seed_entities=seed_entities,
            context_chunks=context_chunks,  # Bağlam için (opsiyonel)
            provenance_graph=provenance_graph.to_dict() if provenance_graph else None,
        )
    
    def _find_seed_entities(
        self,
        query_embedding: List[float],
        k: int = 10,
    ) -> List[str]:
        """Find seed entities by matching query embedding with entity embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Maximum number of entities to return
        
        Returns:
            List of entity IDs sorted by similarity
        """
        entity_scores = []
        
        for node_id, node in self.kg.nodes.items():
            if node.embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            
            if similarity >= self.similarity_threshold:
                entity_scores.append((node_id, similarity))
        
        # Sort by similarity (descending)
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k entity IDs
        return [entity_id for entity_id, _ in entity_scores[:k]]
    
    def _traverse_graph(
        self,
        seed_entities: List[str],
        max_hops: int = 2,
    ) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """Traverse graph from seed entities (BFS).
        
        Args:
            seed_entities: List of seed entity IDs
            max_hops: Maximum number of hops from seed
        
        Returns:
            Tuple of (subgraph, entities, relationships)
        """
        visited = set()
        queue = deque()
        subgraph_nodes = []
        subgraph_edges = []
        entities = []
        relationships = []
        
        # Initialize queue with seed entities
        for entity_id in seed_entities:
            if entity_id in self.kg.nodes:
                queue.append((entity_id, 0))  # (entity_id, hop_count)
                visited.add(entity_id)
        
        # BFS traversal
        while queue:
            current_entity, hop_count = queue.popleft()
            
            if hop_count > max_hops:
                continue
            
            # Add to results
            node = self.kg.get_node(current_entity)
            if node:
                subgraph_nodes.append({
                    "id": node.id,
                    "type": node.type,
                    "properties": node.properties,
                    "metadata": node.metadata,
                })
                entities.append(node.id)
            
            # Explore neighbors
            for edge in self.kg.edges:
                if edge.source == current_entity and edge.target not in visited:
                    queue.append((edge.target, hop_count + 1))
                    visited.add(edge.target)
                    
                    # Add edge
                    edge_dict = {
                        "source": edge.source,
                        "target": edge.target,
                        "relationship_type": edge.relationship_type,
                        "relationship_detail": edge.relationship_detail,
                        "metadata": edge.metadata,
                    }
                    subgraph_edges.append(edge_dict)
                    relationships.append(edge_dict)
                
                elif edge.target == current_entity and edge.source not in visited:
                    queue.append((edge.source, hop_count + 1))
                    visited.add(edge.source)
                    
                    # Add edge (reverse direction)
                    edge_dict = {
                        "source": edge.source,
                        "target": edge.target,
                        "relationship_type": edge.relationship_type,
                        "relationship_detail": edge.relationship_detail,
                        "metadata": edge.metadata,
                    }
                    subgraph_edges.append(edge_dict)
                    relationships.append(edge_dict)
        
        subgraph = {
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
        }
        
        return subgraph, entities, relationships
    
    def _find_relevant_reports(
        self,
        seed_entities: List[str],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find relevant community reports for seed entities.
        
        Args:
            seed_entities: List of seed entity IDs
            k: Maximum number of reports to return
        
        Returns:
            List of community report dictionaries
        """
        relevant_reports = []
        
        # Find clusters containing seed entities
        for cluster_id, cluster in self.kg.clusters.items():
            # Check if any seed entity is in this cluster
            if cluster.node_ids.intersection(set(seed_entities)):
                if cluster_id in self.community_reports:
                    report = self.community_reports[cluster_id]
                    relevant_reports.append(report.to_dict())
        
        # Sort by relevance (number of seed entities in cluster)
        def relevance_score(report_dict: Dict[str, Any]) -> int:
            cluster_id = report_dict.get("cluster_id", "")
            if cluster_id in self.kg.clusters:
                cluster = self.kg.clusters[cluster_id]
                return len(cluster.node_ids.intersection(set(seed_entities)))
            return 0
        
        relevant_reports.sort(key=relevance_score, reverse=True)
        
        return relevant_reports[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity (0-1)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _create_provenance_graph(
        self,
        query: str,
        seed_entities: List[str],
        context_chunks: Optional[List[Dict[str, Any]]],
        relevant_reports: List[Dict[str, Any]],
        entities: List[str],
        relationships: List[Dict[str, Any]],
    ) -> ProvenanceGraph:
        """Create provenance graph for explainable retrieval.
        
        Creates provenance chain: query → chunks → community → summary → answer
        
        Args:
            query: Original query
            seed_entities: Seed entities found
            context_chunks: Context chunks from vector store
            relevant_reports: Relevant community reports
            entities: Entities in subgraph
            relationships: Relationships in subgraph
        
        Returns:
            ProvenanceGraph instance
        """
        nodes = []
        edges = []
        
        # Query node
        query_node = ProvenanceNode(
            id="query",
            type="query",
            label=f"Query: {query[:50]}",
            data={"query": query},
        )
        nodes.append(query_node)
        
        # Chunk nodes (if available)
        chunk_nodes = []
        if context_chunks:
            for idx, chunk in enumerate(context_chunks[:5]):  # Limit to 5 chunks
                chunk_id = f"chunk_{idx}"
                chunk_node = ProvenanceNode(
                    id=chunk_id,
                    type="chunk",
                    label=f"Chunk {idx+1}",
                    data={
                        "chunk_id": chunk.get("chunk_id"),
                        "text": chunk.get("text", "")[:100],
                        "score": chunk.get("score"),
                    },
                    metadata=chunk.get("metadata", {}),
                )
                nodes.append(chunk_node)
                chunk_nodes.append(chunk_id)
                
                # Edge from query to chunk
                edges.append(ProvenanceEdge(
                    source="query",
                    target=chunk_id,
                    type="retrieved_from",
                    label="retrieved",
                    weight=chunk.get("score", 1.0),
                ))
        
        # Community nodes
        community_nodes = []
        for idx, report in enumerate(relevant_reports[:5]):  # Limit to 5 communities
            community_id = f"community_{report.get('cluster_id', idx)}"
            community_node = ProvenanceNode(
                id=community_id,
                type="community",
                label=f"Community {idx+1}",
                data={
                    "cluster_id": report.get("cluster_id"),
                    "summary": report.get("summary", "")[:100],
                },
                metadata=report.get("metadata", {}),
            )
            nodes.append(community_node)
            community_nodes.append(community_id)
            
            # Edge from query to community (via seed entities)
            edges.append(ProvenanceEdge(
                source="query",
                target=community_id,
                type="matched_community",
                label="matched",
                weight=1.0,
            ))
            
            # Summary node
            summary_id = f"summary_{idx}"
            summary_node = ProvenanceNode(
                id=summary_id,
                type="summary",
                label=f"Summary {idx+1}",
                data={
                    "summary": report.get("summary", "")[:200],
                    "themes": report.get("themes", []),
                },
                metadata=report.get("metadata", {}),
            )
            nodes.append(summary_node)
            
            # Edge from community to summary
            edges.append(ProvenanceEdge(
                source=community_id,
                target=summary_id,
                type="summarized_in",
                label="summarized",
                weight=1.0,
            ))
            
            # Connect summary to answer (only first summary)
            if idx == 0:
                answer_id = "answer"
                answer_node = ProvenanceNode(
                    id=answer_id,
                    type="answer",
                    label="Answer",
                    data={
                        "answer": f"Based on {len(relevant_reports)} communities and {len(seed_entities)} entities",
                    },
                )
                nodes.append(answer_node)
                
                edges.append(ProvenanceEdge(
                    source=summary_id,
                    target=answer_id,
                    type="generated_from",
                    label="generated",
                    weight=1.0,
                ))
        
        # Generate answer text
        answer_text = f"Query: {query}\n\n"
        answer_text += f"Found {len(seed_entities)} relevant entities in "
        answer_text += f"{len(relevant_reports)} communities.\n\n"
        
        if relevant_reports:
            answer_text += "Community summaries:\n"
            for report in relevant_reports[:3]:
                answer_text += f"- {report.get('summary', '')}\n"
        
        provenance = ProvenanceGraph(
            nodes=nodes,
            edges=edges,
            query=query,
            answer=answer_text,
            metadata={
                "seed_entities": seed_entities,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )
        
        return provenance


def create_graphrag_retriever(
    embedding_provider: EmbeddingProvider,
    knowledge_graph: EnhancedKG,
    community_reports: Optional[List[CommunityReport]] = None,
    vector_store: Optional[VectorStore] = None,
    use_vector_store_context: bool = False,
    similarity_threshold: float = 0.5,
    max_hops: int = 2,
) -> GraphRAGRetriever:
    """Factory function to create GraphRAG retriever.
    
    Args:
        embedding_provider: Embedding provider
        knowledge_graph: Enhanced knowledge graph with entity embeddings
        community_reports: Optional pre-generated community reports
        vector_store: Optional vector store for context awareness (çok yüksek veri için)
        use_vector_store_context: Whether to use vector store for context awareness
        similarity_threshold: Minimum similarity for seed entity matching
        max_hops: Maximum graph traversal hops
    
    Returns:
        GraphRAGRetriever instance
    """
    return GraphRAGRetriever(
        embedding_provider=embedding_provider,
        knowledge_graph=knowledge_graph,
        community_reports=community_reports,
        vector_store=vector_store,
        use_vector_store_context=use_vector_store_context,
        similarity_threshold=similarity_threshold,
        max_hops=max_hops,
    )

