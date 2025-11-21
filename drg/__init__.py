"""DRG - Declarative Relationship Generation"""
__version__ = "0.1.0a0"

from .schema import Entity, Relation, DRGSchema
from .extract import extract_typed, extract_triples, KGExtractor
from .graph import KG
from .optimize import (
    refine_triples,
    optimize_extractor,
    merge_entities,
    merge_relations,
    kg_metric
)
from .mcp_api import build, build_from_file

__all__ = [
    "Entity",
    "Relation", 
    "DRGSchema",
    "extract_typed",
    "extract_triples",
    "KGExtractor",
    "KG",
    "refine_triples",
    "optimize_extractor",
    "merge_entities",
    "merge_relations",
    "kg_metric",
    "build",
    "build_from_file",
]
