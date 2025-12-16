"""
Knowledge Graph Module

This module provides:
- Schema generation (dataset-agnostic)
- Relationship modeling (enriched, GraphRAG format)
- Knowledge graph core (modular monolith)
- Visualization (Mermaid, PyVis)
- Community reports (GraphRAG style)
"""

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

__all__ = [
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
]

