"""
Visualization Adapter - JSON Export for JS Graph Libraries

Converts internal graph structures and answer provenance chains into
JSON formats consumable by JavaScript graph visualization libraries
(Cytoscape.js, vis-network, D3.js, etc.)
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os
from collections import Counter, defaultdict

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
    """Complete provenance graph for explainable analysis/query flows."""
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

    @staticmethod
    def _is_hubproxy_id(node_id: str) -> bool:
        """Return True if node id corresponds to a hub-split proxy node."""
        return isinstance(node_id, str) and node_id.startswith("hubproxy::")

    def _flatten_hubproxy_view(self, kg: EnhancedKG) -> tuple[Dict[str, KGNode], List[KGEdge]]:
        """Create a visualization-only view with hub-proxy nodes removed.

        Some example KG exports (e.g., `outputs/1example_kg.json`) are already hub-split
        at the KG level (they contain `hubproxy::...` nodes and re-routed edges).
        When the UI's `hub_split` toggle is OFF, we still want a clean graph without
        proxy nodes/ids.

        This function rebuilds original edges from `edge.metadata["triple"]` when
        available and drops structural connector edges.
        """
        edges_out: List[KGEdge] = []
        seen: set[tuple[str, str, str, str]] = set()

        for edge in kg.edges:
            md = edge.metadata or {}
            proxy_kind = md.get("proxy_kind")

            # Drop structural connector edges (hub -> proxy) entirely.
            if proxy_kind == "hub_proxy_connector":
                continue

            touches_proxy = self._is_hubproxy_id(edge.source) or self._is_hubproxy_id(edge.target)
            looks_proxy_edge = touches_proxy or proxy_kind == "hub_split_edge"

            if looks_proxy_edge:
                triple = md.get("triple")
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    src, rel, dst = triple
                    if isinstance(src, str) and isinstance(rel, str) and isinstance(dst, str):
                        new_md = dict(md)
                        new_md.pop("proxy_kind", None)
                        new_md.pop("hub", None)
                        new_md["flattened_from_proxy"] = True
                        rebuilt = KGEdge(
                            source=src,
                            target=dst,
                            relationship_type=rel,
                            relationship_detail=edge.relationship_detail,
                            metadata=new_md,
                            start_time=edge.start_time,
                            end_time=edge.end_time,
                            confidence=edge.confidence,
                            is_negated=edge.is_negated,
                        )
                        k = (rebuilt.source, rebuilt.target, rebuilt.relationship_type, rebuilt.relationship_detail)
                        if k not in seen:
                            edges_out.append(rebuilt)
                            seen.add(k)
                        continue

                # If it's proxy-related but we can't rebuild it, drop it to avoid leaking hubproxy ids.
                if touches_proxy:
                    continue

            # Keep normal edges as-is
            k = (edge.source, edge.target, edge.relationship_type, edge.relationship_detail)
            if k not in seen:
                edges_out.append(edge)
                seen.add(k)

        connected: set[str] = set()
        for e in edges_out:
            connected.add(e.source)
            connected.add(e.target)

        nodes_out: Dict[str, KGNode] = {}
        for nid in connected:
            node = kg.nodes.get(nid)
            if node is None:
                node = KGNode(id=nid, type=None)
            nodes_out[nid] = node

        return nodes_out, edges_out
    
    def kg_to_cytoscape(
        self,
        kg: Optional[EnhancedKG] = None,
        *,
        hub_split: Optional[bool] = None,
        hub_split_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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

        # Resolve hub split defaults early (we need it to decide whether to flatten proxy nodes)
        if hub_split is None:
            hub_split = os.getenv("DRG_UI_HUB_SPLIT", "0").strip().lower() in {"1", "true", "yes", "y"}
        if hub_split_threshold is None:
            try:
                hub_threshold = int(os.getenv("DRG_UI_HUB_SPLIT_THRESHOLD", "10"))
            except Exception:
                hub_threshold = 10
        else:
            hub_threshold = int(hub_split_threshold)

        # Use a visualization-only flattened view when hub_split is OFF, so the UI never
        # shows `hubproxy::...` ids from already hub-split KG exports.
        nodes_view: Dict[str, KGNode]
        edges_view: List[KGEdge]
        if not hub_split:
            nodes_view, edges_view = self._flatten_hubproxy_view(kg)
        else:
            nodes_view, edges_view = kg.nodes, list(kg.edges)

        elements: List[Dict[str, Any]] = []

        # First, collect all nodes that have at least one edge (connected nodes)
        connected_node_ids: set[str] = set()
        for edge in edges_view:
            connected_node_ids.add(edge.source)
            connected_node_ids.add(edge.target)

        # Add only connected nodes (nodes with at least one edge)
        for node in nodes_view.values():
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
            connection_count = sum(1 for edge in edges_view if edge.source == node.id or edge.target == node.id)
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
        
        # Optionally "split" high-degree hub nodes into relation-specific proxy nodes for the UI.
        # This is visualization-only: it does NOT change the stored KG, and avoids "single-node centered"
        # layouts for naturally hub-like documents (e.g., company-product lists).
        # Default OFF in UI; user can enable via query param or env.
        deg = Counter()
        incident: Dict[str, List[KGEdge]] = defaultdict(list)
        for e in edges_view:
            deg[e.source] += 1
            deg[e.target] += 1
            incident[e.source].append(e)
            incident[e.target].append(e)

        hubs = {n for n, d in deg.items() if d >= hub_threshold} if hub_split else set()

        # Track which original edges are replaced by proxy edges
        replaced_edge_keys = set()

        # Map node_id -> community_id (best-effort, first match)
        node_to_community: Dict[str, Optional[str]] = {}
        if kg.clusters:
            for node_id in connected_node_ids:
                cid = None
                for cluster_id, cluster in kg.clusters.items():
                    if node_id in cluster.node_ids:
                        cid = cluster_id
                        break
                node_to_community[node_id] = cid

        # Add proxy nodes (one per hub x relationship_type group)
        proxy_nodes_by_hub_rel: Dict[tuple[str, str], str] = {}
        if hubs:
            for hub in sorted(hubs, key=lambda s: s.lower()):
                # Group incident edges by relation type
                rel_groups: Dict[str, List[KGEdge]] = defaultdict(list)
                for e in incident.get(hub, []):
                    rel_groups[e.relationship_type].append(e)

                for rel in sorted(rel_groups.keys(), key=lambda s: s.lower()):
                    proxy_id = f"hubproxy::{hub}::{rel}"
                    proxy_nodes_by_hub_rel[(hub, rel)] = proxy_id

                    # Create a visible proxy node (acts like a "relation fan-out" point)
                    hub_comm = node_to_community.get(hub)
                    proxy_data = {
                        "data": {
                            "id": proxy_id,
                            "label": rel,
                            "type": "HubProxy",
                            "properties": {},
                            "metadata": {
                                "hub": hub,
                                "relationship_type": rel,
                                "proxy_kind": "hub_relation_proxy",
                                "edge_count": len(rel_groups[rel]),
                            },
                            "weight": len(rel_groups[rel]),
                            "connection_count": len(rel_groups[rel]),
                        },
                        "classes": ["HubProxy"],
                        "style": {
                            "background-color": "#CBD5E1",  # slate-300
                            "label": rel,
                            "shape": "round-rectangle",
                            "text-wrap": "wrap",
                            "text-max-width": "120px",
                            "font-size": "9px",
                            "width": 18,
                            "height": 18,
                            "border-width": 1,
                            "border-color": "#94A3B8",  # slate-400
                        },
                    }
                    if hub_comm:
                        proxy_data["data"]["community_id"] = hub_comm
                        proxy_data["data"]["community"] = hub_comm
                    elements.append(proxy_data)

                    # Connect hub -> proxy with a lightweight edge (no evidence; it's structural)
                    hub_proxy_edge = {
                        "data": {
                            "id": f"{hub}--{proxy_id}",
                            "source": hub,
                            "target": proxy_id,
                            "label": rel,
                            "relationship_type": rel,
                            "relationship_detail": "",
                            "relationship_description": "UI structural edge for hub splitting.",
                            "weight": 0.1,
                            "metadata": {"proxy_kind": "hub_proxy_connector"},
                        },
                        "style": {
                            "width": 1,
                            "line-color": "#CBD5E1",
                            "label": "",
                            "line-style": "dashed",
                        },
                    }
                    elements.append(hub_proxy_edge)

        def _edge_key(edge: KGEdge) -> tuple[str, str, str, str]:
            return (edge.source, edge.target, edge.relationship_type, edge.relationship_detail)

        # Add edges (with optional hub splitting)
        for edge in edges_view:
            if hubs and (edge.source in hubs or edge.target in hubs):
                k = _edge_key(edge)
                if k in replaced_edge_keys:
                    continue

                # Pick one hub to "own" the split for this edge (prefer source for determinism)
                hub = edge.source if edge.source in hubs else edge.target
                proxy_id = proxy_nodes_by_hub_rel.get((hub, edge.relationship_type))
                if not proxy_id:
                    # Shouldn't happen, but fall back to rendering the original edge
                    pass
                else:
                    replaced_edge_keys.add(k)
                    if edge.source == hub:
                        new_source = proxy_id
                        new_target = edge.target
                    else:
                        new_source = edge.source
                        new_target = proxy_id

                    # Keep original edge semantics/details on the proxy edge
                    weight = edge.metadata.get("weight", 1.0)
                    if "confidence" in edge.metadata:
                        weight = edge.metadata["confidence"]
                    color = self._get_edge_color(edge.relationship_type)
                    edge_label = edge.relationship_type

                    edge_data = {
                        "data": {
                            "id": f"{new_source}-{new_target}-{edge.relationship_type}",
                            "source": new_source,
                            "target": new_target,
                            "label": edge_label,
                            "relationship_type": edge.relationship_type,
                            "relationship_detail": edge.relationship_detail,
                            "relationship_description": (
                                edge.metadata.get("relationship_description")
                                or edge.metadata.get("description")
                                or ""
                            ),
                            "weight": float(weight),
                            "metadata": {**edge.metadata, "proxy_kind": "hub_split_edge", "hub": hub},
                        },
                        "style": {
                            "width": max(3, min(10, float(weight) * 5)),
                            "line-color": color,
                            "label": edge_label,
                        },
                    }
                    elements.append(edge_data)
                    continue

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
                    "relationship_description": (
                        edge.metadata.get("relationship_description")
                        or edge.metadata.get("description")
                        or ""
                    ),
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
        *,
        hub_split: Optional[bool] = None,
        hub_split_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convert KG with communities to Cytoscape.js format with color coding.
        
        Args:
            kg: EnhancedKG instance (uses self.kg if None)
            community_reports: Optional list of community reports
        
        Returns:
            List of Cytoscape elements with community color coding
        """
        # Use the filtered version (only connected nodes)
        elements = self.kg_to_cytoscape(
            kg,
            hub_split=hub_split,
            hub_split_threshold=hub_split_threshold,
        )
        
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

