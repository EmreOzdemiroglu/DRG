"""DRG - Declarative Relationship Generation"""
__version__ = "0.1.0a0"

from .schema import (
    Entity,
    Relation,
    DRGSchema,
    EntityType,
    EntityGroup,
    PropertyGroup,
    RelationGroup,
    EnhancedDRGSchema,
)
from .extract import extract_typed, extract_triples, KGExtractor
from .graph import KG

__all__ = [
    # Legacy classes
    "Entity",
    "Relation",
    "DRGSchema",
    # Enhanced schema classes
    "EntityType",
    "EntityGroup",
    "PropertyGroup",
    "RelationGroup",
    "EnhancedDRGSchema",
    # Extraction
    "extract_typed",
    "extract_triples",
    "KGExtractor",
    # Graph
    "KG",
]
