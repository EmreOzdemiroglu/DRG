"""
Visualization Adapter - JSON Export for JS Graph Libraries

Converts internal graph structures and answer provenance chains into
JSON formats consumable by JavaScript graph visualization libraries
(Cytoscape.js, vis-network, D3.js, etc.)
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .kg_core import EnhancedKG, KGNode, KGEdge, Cluster
from .community_report import CommunityReport

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceNode:
    """Node in a provenance chain."""
    id: str
    type: str  # "query", "chunk", "community", "summary", "answer"
    label: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceEdge:
    """Edge in a provenance chain."""
    source: str
    target: str
    type: str  # "retrieved_from", "summarized_in", "generated_from"
    label: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceGraph:
    """Complete provenance graph for explainable retrieval."""
    nodes: List[ProvenanceNode]
    edges: List[ProvenanceEdge]
    query: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "data": node.data,
                    "metadata": node.metadata,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "label": edge.label,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
            "metadata": self.metadata,
        }


class VisualizationAdapter:
    """
    Adapter for converting KG structures to JS graph library formats.
    
    Supports:
    - Cytoscape.js format
    - vis-network format
    - D3.js format
    - Generic JSON format
    """
    
    def __init__(self, kg: Optional[EnhancedKG] = None):
        """Initialize visualization adapter.
        
        Args:
            kg: Optional EnhancedKG instance
        """
        self.kg = kg
    
    def kg_to_cytoscape(self, kg: Optional[EnhancedKG] = None) -> List[Dict[str, Any]]:
        """Convert EnhancedKG to Cytoscape.js format.
        
        Cytoscape format: List of nodes and edges with data attributes.
        
        Args:
            kg: EnhancedKG instance (uses self.kg if None)
        
        Returns:
            List of Cytoscape elements (nodes + edges)
        """
        kg = kg or self.kg
        if kg is None:
            raise ValueError("No knowledge graph provided")
        
        elements = []
        
        # First, collect all nodes that have at least one edge (connected nodes)
        connected_node_ids = set()
        for edge in kg.edges:
            connected_node_ids.add(edge.source)
            connected_node_ids.add(edge.target)
        
        # Add only connected nodes (nodes with at least one edge)
        for node in kg.nodes.values():
            # Skip isolated nodes (nodes without any edges)
            if node.id not in connected_node_ids:
                continue
            # Determine node color based on type
            color = self._get_node_color(node.type)
            
            # Get community/cluster ID if available
            community_id = None
            for cluster_id, cluster in kg.clusters.items():
                if node.id in cluster.node_ids:
                    community_id = cluster_id
                    break
            
            # Calculate node weight based on connections
            connection_count = sum(1 for edge in kg.edges if edge.source == node.id or edge.target == node.id)
            node_weight = max(1, min(10, connection_count))
            
            # Create clean label - just the ID
            node_label = node.id
            
            node_data = {
                "data": {
                    "id": node.id,
                    "label": node_label,
                    "type": node.type or "Unknown",
                    "properties": node.properties,
                    "metadata": node.metadata,
                    "weight": node_weight,
                    "connection_count": connection_count,
                },
                "classes": [node.type or "entity"] if node.type else ["entity"],
                "style": {
                    "background-color": color,
                    "label": node_label,
                },
            }
            
            if community_id:
                node_data["data"]["community_id"] = community_id
                node_data["data"]["community"] = community_id
            
            elements.append(node_data)
        
        # Add edges
        for edge in kg.edges:
            # Get edge weight from metadata
            weight = edge.metadata.get("weight", 1.0)
            if "confidence" in edge.metadata:
                weight = edge.metadata["confidence"]
            
            # Determine edge color based on relationship type
            color = self._get_edge_color(edge.relationship_type)
            
            # Create edge label - just relationship type (detail is shown in tooltip)
            edge_label = edge.relationship_type
            
            edge_data = {
                "data": {
                    "id": f"{edge.source}-{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge_label,
                    "relationship_type": edge.relationship_type,
                    "relationship_detail": edge.relationship_detail,
                    "weight": float(weight),
                    "metadata": edge.metadata,
                },
                "style": {
                    "width": max(3, min(10, weight * 5)),  # Scale width by weight (3-10px)
                    "line-color": color,
                    "label": edge_label,
                },
            }
            
            elements.append(edge_data)
        
        return elements
    
    def kg_to_vis_network(self, kg: Optional[EnhancedKG] = None) -> Dict[str, Any]:
        """Convert EnhancedKG to vis-network format.
        
        vis-network format: Dictionary with "nodes" and "edges" arrays.
        
        Args:
            kg: EnhancedKG instance (uses self.kg if None)
        
        Returns:
            Dictionary with "nodes" and "edges" keys
        """
        kg = kg or self.kg
        if kg is None:
            raise ValueError("No knowledge graph provided")
        
        nodes = []
        edges = []
        
        # First, collect all nodes that have at least one edge (connected nodes)
        connected_node_ids = set()
        for edge in kg.edges:
            connected_node_ids.add(edge.source)
            connected_node_ids.add(edge.target)
        
        # Add only connected nodes (nodes with at least one edge)
        for node in kg.nodes.values():
            # Skip isolated nodes (nodes without any edges)
            if node.id not in connected_node_ids:
                continue
            color = self._get_node_color(node.type)
            
            # Get community/cluster ID
            community_id = None
            for cluster_id, cluster in kg.clusters.items():
                if node.id in cluster.node_ids:
                    community_id = cluster_id
                    break
            
            node_data = {
                "id": node.id,
                "label": node.id,
                "title": f"Type: {node.type or 'Unknown'}\nID: {node.id}",
                "color": color,
                "type": node.type or "Unknown",
                "properties": node.properties,
                "metadata": node.metadata,
            }
            
            if community_id:
                node_data["group"] = community_id
                node_data["community_id"] = community_id
            
            nodes.append(node_data)
        
        # Add edges
        for edge in kg.edges:
            weight = edge.metadata.get("weight", 1.0)
            if "confidence" in edge.metadata:
                weight = edge.metadata["confidence"]
            
            color = self._get_edge_color(edge.relationship_type)
            
            edge_data = {
                "id": f"{edge.source}-{edge.target}",
                "from": edge.source,
                "to": edge.target,
                "label": edge.relationship_type,
                "title": f"{edge.relationship_type}\n{edge.relationship_detail}",
                "value": float(weight),
                "color": {"color": color},
                "relationship_type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                "metadata": edge.metadata,
            }
            
            edges.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges,
        }
    
    def kg_to_d3_json(self, kg: Optional[EnhancedKG] = None) -> Dict[str, Any]:
        """Convert EnhancedKG to D3.js force-directed graph format.
        
        D3 format: Dictionary with "nodes" and "links" arrays.
        
        Args:
            kg: EnhancedKG instance (uses self.kg if None)
        
        Returns:
            Dictionary with "nodes" and "links" keys
        """
        kg = kg or self.kg
        if kg is None:
            raise ValueError("No knowledge graph provided")
        
        nodes = []
        links = []
        
        # First, collect all nodes that have at least one edge (connected nodes)
        connected_node_ids = set()
        for edge in kg.edges:
            connected_node_ids.add(edge.source)
            connected_node_ids.add(edge.target)
        
        # Filter to only connected nodes for indexing
        connected_nodes_list = [(node_id, node) for node_id, node in kg.nodes.items() if node_id in connected_node_ids]
        node_index = {node_id: idx for idx, (node_id, _) in enumerate(connected_nodes_list)}
        
        # Add only connected nodes
        for idx, (node_id, node) in enumerate(connected_nodes_list):
            color = self._get_node_color(node.type)
            
            # Get community/cluster ID
            community_id = None
            for cluster_id, cluster in kg.clusters.items():
                if node.id in cluster.node_ids:
                    community_id = cluster_id
                    break
            
            node_data = {
                "id": node.id,
                "name": node.id,
                "type": node.type or "Unknown",
                "color": color,
                "group": community_id or 0,
                "properties": node.properties,
                "metadata": node.metadata,
            }
            
            if community_id:
                node_data["community_id"] = community_id
            
            nodes.append(node_data)
        
        # Add links
        for edge in kg.edges:
            if edge.source not in node_index or edge.target not in node_index:
                continue
            
            weight = edge.metadata.get("weight", 1.0)
            if "confidence" in edge.metadata:
                weight = edge.metadata["confidence"]
            
            link_data = {
                "source": node_index[edge.source],
                "target": node_index[edge.target],
                "value": float(weight),
                "type": edge.relationship_type,
                "relationship_detail": edge.relationship_detail,
                "metadata": edge.metadata,
            }
            
            links.append(link_data)
        
        return {
            "nodes": nodes,
            "links": links,
        }
    
    def communities_to_cytoscape(
        self,
        kg: Optional[EnhancedKG] = None,
        community_reports: Optional[List[CommunityReport]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert KG with communities to Cytoscape.js format with color coding.
        
        Args:
            kg: EnhancedKG instance (uses self.kg if None)
            community_reports: Optional list of community reports
        
        Returns:
            List of Cytoscape elements with community color coding
        """
        # Use the filtered version (only connected nodes)
        elements = self.kg_to_cytoscape(kg)
        
        # Add community color coding
        kg = kg or self.kg
        if kg is None:
            return elements
        
        # Map community IDs to colors
        community_colors = {}
        for idx, cluster_id in enumerate(kg.clusters.keys()):
            community_colors[cluster_id] = self._get_community_color(idx)
        
        # Update node colors based on community
        for element in elements:
            if "data" in element and "community_id" in element["data"]:
                community_id = element["data"]["community_id"]
                if community_id in community_colors:
                    element["style"]["background-color"] = community_colors[community_id]
        
        return elements
    
    def provenance_to_cytoscape(
        self,
        provenance: ProvenanceGraph,
    ) -> List[Dict[str, Any]]:
        """Convert provenance graph to Cytoscape.js format.
        
        Args:
            provenance: ProvenanceGraph instance
        
        Returns:
            List of Cytoscape elements representing provenance chain
        """
        elements = []
        
        # Add nodes
        type_colors = {
            "query": "#FF6B6B",
            "chunk": "#4ECDC4",
            "community": "#FFE66D",
            "summary": "#95E1D3",
            "answer": "#F38181",
        }
        
        for node in provenance.nodes:
            color = type_colors.get(node.type, "#A8A8A8")
            
            node_data = {
                "data": {
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "data": node.data,
                    "metadata": node.metadata,
                },
                "classes": [node.type],
                "style": {
                    "background-color": color,
                    "label": node.label,
                },
            }
            
            elements.append(node_data)
        
        # Add edges
        for edge in provenance.edges:
            edge_data = {
                "data": {
                    "id": f"{edge.source}-{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "type": edge.type,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                },
                "style": {
                    "width": max(1, min(5, edge.weight * 3)),
                    "label": edge.label,
                },
            }
            
            elements.append(edge_data)
        
        return elements
    
    def provenance_to_json(self, provenance: ProvenanceGraph) -> Dict[str, Any]:
        """Convert provenance graph to generic JSON format.
        
        Args:
            provenance: ProvenanceGraph instance
        
        Returns:
            Dictionary with provenance graph data
        """
        return {
            "query": provenance.query,
            "answer": provenance.answer,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "label": node.label,
                    "data": node.data,
                    "metadata": node.metadata,
                }
                for node in provenance.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "label": edge.label,
                    "weight": edge.weight,
                    "metadata": edge.metadata,
                }
                for edge in provenance.edges
            ],
            "metadata": provenance.metadata,
        }
    
    def _get_node_color(self, node_type: Optional[str]) -> str:
        """Get color for node type."""
        type_colors = {
            "Person": "#FF6B6B",
            "Location": "#4ECDC4",
            "Event": "#FFE66D",
            "Organization": "#95E1D3",
            "Product": "#F38181",
            "Company": "#95E1D3",
            "default": "#A8A8A8",
        }
        return type_colors.get(node_type, type_colors["default"])
    
    def _get_edge_color(self, relationship_type: str) -> str:
        """Get color for relationship type."""
        type_colors = {
            "influences": "#FF6B6B",
            "caused_by": "#4ECDC4",
            "located_at": "#95E1D3",
            "collaborates_with": "#FFE66D",
            "works_with": "#FFE66D",
            "default": "#CCCCCC",
        }
        return type_colors.get(relationship_type, type_colors["default"])
    
    def _get_community_color(self, index: int) -> str:
        """Get color for community by index.
        
        Uses a color palette that provides good visual distinction.
        """
        colors = [
            "#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181",
            "#A8E6CF", "#FFD3B6", "#FFAAA5", "#FF8B94", "#C7CEEA",
        ]
        return colors[index % len(colors)]

