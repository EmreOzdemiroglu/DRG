"""DRG (Dynamic Retrieval Graph) search algorithms."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

from ..graph import KG
from ..graph.kg_core import EnhancedKG
from ..embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


class DRGSearch:
    """DRG search with graph traversal algorithms (GraphRAG-compatible)."""
    
    def __init__(
        self,
        knowledge_graph: KG,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize DRG search.
        
        Args:
            knowledge_graph: Knowledge graph to search (can be KG or EnhancedKG)
            embedding_provider: Embedding provider for semantic similarity (seed entity matching)
        """
        self.kg = knowledge_graph
        self.embedding_provider = embedding_provider
        # Check if it's EnhancedKG
        self.is_enhanced = isinstance(knowledge_graph, EnhancedKG)
    
    def bfs_search(
        self,
        seed_entities: List[str],
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> List[Dict[str, Any]]:
        """Breadth-first search from seed entities.
        
        Args:
            seed_entities: List of seed entity names
            max_hops: Maximum number of hops from seed
            max_nodes: Maximum number of nodes to explore
        
        Returns:
            List of nodes with their hop distance and metadata
        """
        visited = set()
        queue = deque()
        results = []
        
        # Initialize queue with seed entities
        for entity in seed_entities:
            if self.is_enhanced:
                if self.kg.get_node(entity) is not None:
                    queue.append((entity, 0))
                    visited.add(entity)
            else:
                if entity in self.kg.nodes:
                    queue.append((entity, 0))
                    visited.add(entity)
        
        while queue and len(results) < max_nodes:
            current_entity, hop_count = queue.popleft()
            
            if hop_count > max_hops:
                continue
            
            # Add to results
            if self.is_enhanced:
                node = self.kg.get_node(current_entity)
                node_data = node.to_dict() if node else {}
            else:
                node_data = self.kg.nodes.get(current_entity, {})
            
            results.append({
                "entity": current_entity,
                "hop_count": hop_count,
                "type": node_data.get("type"),
                "metadata": node_data,
            })
            
            # Explore neighbors
            for edge in self.kg.edges:
                source, relation, target = edge
                if source == current_entity and target not in visited:
                    queue.append((target, hop_count + 1))
                    visited.add(target)
                elif target == current_entity and source not in visited:
                    queue.append((source, hop_count + 1))
                    visited.add(source)
        
        return results
    
    def dfs_search(
        self,
        seed_entities: List[str],
        query_embedding: Optional[List[float]] = None,
        max_depth: int = 3,
        max_nodes: int = 50,
        similarity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Depth-first search with semantic guidance.
        
        Args:
            seed_entities: List of seed entity names
            query_embedding: Optional query embedding for semantic guidance
            max_depth: Maximum depth to explore
            max_nodes: Maximum number of nodes to explore
            similarity_threshold: Minimum similarity to continue exploring branch
        
        Returns:
            List of nodes with their depth and metadata
        """
        visited = set()
        results = []
        
        def dfs_recursive(entity: str, depth: int):
            """Recursive DFS helper."""
            if depth > max_depth or len(results) >= max_nodes or entity in visited:
                return
            
            visited.add(entity)
            
            # Check if entity exists
            if self.is_enhanced:
                node = self.kg.get_node(entity)
                if node is None:
                    return
            else:
                if entity not in self.kg.nodes:
                    return
            
            # Check semantic similarity if query embedding provided
            if query_embedding:
                if self.is_enhanced:
                    node = self.kg.get_node(entity)
                    node_embedding = node.embedding if node else None
                else:
                    node_data = self.kg.nodes.get(entity, {})
                    node_embedding = node_data.get("embedding")
                
                if node_embedding:
                    similarity = self._cosine_similarity(query_embedding, node_embedding)
                    if similarity < similarity_threshold and depth > 0:
                        # Don't explore this branch if similarity is too low
                        return
                else:
                    # No embedding, use default threshold
                    if depth > 1:
                        return
            
            # Add to results
            if self.is_enhanced:
                node = self.kg.get_node(entity)
                node_data = node.to_dict() if node else {}
            else:
                node_data = self.kg.nodes.get(entity, {})
            
            results.append({
                "entity": entity,
                "depth": depth,
                "type": node_data.get("type"),
                "metadata": node_data,
            })
            
            # Explore neighbors (prioritize by semantic similarity if available)
            neighbors = []
            for edge in self.kg.edges:
                source, relation, target = edge
                if source == entity and target not in visited:
                    neighbors.append((target, relation))
                elif target == entity and source not in visited:
                    neighbors.append((source, relation))
            
            # Sort neighbors by semantic similarity if query embedding available
            if query_embedding:
                neighbor_scores = []
                for neighbor_entity, relation in neighbors:
                    if self.is_enhanced:
                        neighbor_node = self.kg.get_node(neighbor_entity)
                        neighbor_embedding = neighbor_node.embedding if neighbor_node else None
                    else:
                        neighbor_data = self.kg.nodes.get(neighbor_entity, {})
                        neighbor_embedding = neighbor_data.get("embedding")
                    
                    if neighbor_embedding:
                        score = self._cosine_similarity(query_embedding, neighbor_embedding)
                        neighbor_scores.append((score, neighbor_entity, relation))
                    else:
                        neighbor_scores.append((0.0, neighbor_entity, relation))
                
                # Sort by score (descending)
                neighbor_scores.sort(reverse=True)
                neighbors = [(entity, rel) for _, entity, rel in neighbor_scores]
            
            # Recursively explore neighbors
            for neighbor_entity, relation in neighbors:
                dfs_recursive(neighbor_entity, depth + 1)
        
        # Start DFS from each seed entity
        for seed in seed_entities:
            if seed in self.kg.nodes:
                dfs_recursive(seed, 0)
        
        return results
    
    def weighted_search(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None,
        max_hops: int = 2,
        k: int = 10,
        alpha: float = 0.7,  # Weight for semantic similarity
        beta: float = 0.3,   # Weight for graph proximity
    ) -> List[Dict[str, Any]]:
        """Weighted search combining semantic similarity and graph distance (GraphRAG-compatible).
        
        Args:
            query: Query text
            seed_entities: Optional seed entities (found via embedding matching if not provided)
            max_hops: Maximum graph hops
            k: Number of results to return
            alpha: Weight for semantic similarity (0-1)
            beta: Weight for graph proximity (0-1, should be 1-alpha)
        
        Returns:
            List of nodes with combined scores
        """
        # Embed query
        query_embedding = self.embedding_provider.embed(query)
        
        # Extract seed entities if not provided (GraphRAG: match entity embeddings)
        if seed_entities is None:
            if self.is_enhanced:
                # Use embedding-based seed entity finding (GraphRAG approach)
                seed_entities = self._find_seed_entities_by_embedding(query_embedding, k=5)
            else:
                # Fallback to text-based extraction
                seed_entities = self._extract_entities_from_query(query)
        
        # BFS to get candidate nodes
        candidate_nodes = self.bfs_search(seed_entities, max_hops=max_hops)
        
        # Score each candidate
        scored_nodes = []
        for node in candidate_nodes:
            entity = node["entity"]
            hop_count = node["hop_count"]
            
            # Get node embedding if available
            if self.is_enhanced:
                node = self.kg.get_node(entity)
                node_embedding = node.embedding if node else None
            else:
                node_data = self.kg.nodes.get(entity, {})
                node_embedding = node_data.get("embedding")
            
            # Calculate semantic similarity
            semantic_score = 0.0
            if node_embedding:
                # Cosine similarity
                semantic_score = self._cosine_similarity(query_embedding, node_embedding)
            else:
                # Fallback: use text similarity if no embedding
                semantic_score = 0.5  # Default score
            
            # Calculate graph proximity score
            proximity_score = 1.0 / (1.0 + hop_count) if hop_count > 0 else 1.0
            
            # Combined score
            combined_score = alpha * semantic_score + beta * proximity_score
            
            scored_nodes.append({
                **node_result,
                "semantic_score": semantic_score,
                "proximity_score": proximity_score,
                "combined_score": combined_score,
            })
        
        # Sort by combined score and return top-k
        scored_nodes.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_nodes[:k]
    
    def _find_seed_entities_by_embedding(
        self,
        query_embedding: List[float],
        k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[str]:
        """Find seed entities by matching query embedding with entity embeddings (GraphRAG).
        
        Args:
            query_embedding: Query embedding vector
            k: Maximum number of entities to return
            similarity_threshold: Minimum similarity threshold
        
        Returns:
            List of entity IDs sorted by similarity
        """
        if not self.is_enhanced:
            return []
        
        entity_scores = []
        for node_id, node in self.kg.nodes.items():
            if node.embedding is None:
                continue
            
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            if similarity >= similarity_threshold:
                entity_scores.append((node_id, similarity))
        
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in entity_scores[:k]]
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from query (simplified fallback).
        
        Args:
            query: Query text
        
        Returns:
            List of potential entity names
        """
        # Simple heuristic: look for capitalized words
        # In production, use proper NER
        words = query.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1]
        return entities[:5]  # Limit to 5 entities
    
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


def create_drg_search(
    knowledge_graph: KG,
    embedding_provider: EmbeddingProvider,
) -> DRGSearch:
    """Factory function to create DRG search.
    
    Args:
        knowledge_graph: Knowledge graph
        embedding_provider: Embedding provider
    
    Returns:
        DRGSearch instance
    """
    return DRGSearch(
        knowledge_graph=knowledge_graph,
        embedding_provider=embedding_provider,
    )

