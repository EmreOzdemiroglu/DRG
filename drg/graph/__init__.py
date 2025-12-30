"""
Knowledge Graph Module

This module provides:
- Schema generation (dataset-agnostic)
- Relationship modeling (enriched format)
- Knowledge graph core (modular monolith)
- Visualization (Mermaid, PyVis)
- Community reports
"""

# Legacy KG class (from graph.py)
from typing import List, Dict, Any, Tuple
import json


class KG:
    """Simple Knowledge Graph class."""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []

    @classmethod
    def from_typed(cls, entities_typed: List[Tuple[str, str]], triples: List[Tuple[str, str, str]]):
        kg = cls()
        for name, etype in entities_typed:
            kg.nodes.setdefault(name, {"type": etype})
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    @classmethod
    def from_triples(cls, triples: List[Tuple[str, str, str]]):
        kg = cls()
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    def to_json(self, indent: int = 2) -> str:
        data = {
            "nodes": [{"id": n, **attr} for n, attr in self.nodes.items()],
            "edges": [{"source": s, "type": r, "target": o} for s, r, o in self.edges],
        }
        return json.dumps(data, indent=indent)


# Schema Generator
from .schema_generator import (
    PropertyDefinition,
    EntityClassDefinition,
    DatasetAgnosticSchemaGenerator,
    create_default_schema,
)

# Relationship Model
from .relationship_model import (
    RelationshipType,
    EnrichedRelationship,
    RelationshipTypeClassifier,
    create_enriched_relationship,
    RELATIONSHIP_CATEGORIES,
)

# KG Core
from .kg_core import (
    KGNode,
    KGEdge,
    Cluster,
    EnhancedKG,
)

# Visualization
from .visualization import (
    KGVisualizer,
    DEFAULT_NODE_COLORS,
    DEFAULT_EDGE_COLORS,
)

# Community Report
from .community_report import (
    CommunityReport,
    CommunityReportGenerator,
)

# Neo4j Exporter
from .neo4j_exporter import (
    Neo4jConfig,
    Neo4jExporter,
)

# Visualization Adapter
from .visualization_adapter import (
    ProvenanceNode,
    ProvenanceEdge,
    ProvenanceGraph,
    VisualizationAdapter,
)

# Hub mitigation (export-time graph shaping)
from .hub_mitigation import apply_hub_relation_proxy_split

__all__ = [
    # Legacy KG
    "KG",
    # Schema Generator
    "PropertyDefinition",
    "EntityClassDefinition",
    "DatasetAgnosticSchemaGenerator",
    "create_default_schema",
    # Relationship Model
    "RelationshipType",
    "EnrichedRelationship",
    "RelationshipTypeClassifier",
    "create_enriched_relationship",
    "RELATIONSHIP_CATEGORIES",
    # KG Core
    "KGNode",
    "KGEdge",
    "Cluster",
    "EnhancedKG",
    # Visualization
    "KGVisualizer",
    "DEFAULT_NODE_COLORS",
    "DEFAULT_EDGE_COLORS",
    # Community Report
    "CommunityReport",
    "CommunityReportGenerator",
    # Neo4j Exporter
    "Neo4jConfig",
    "Neo4jExporter",
    # Visualization Adapter
    "ProvenanceNode",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "VisualizationAdapter",
    # Hub mitigation
    "apply_hub_relation_proxy_split",
]

