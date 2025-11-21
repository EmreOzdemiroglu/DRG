"""
MCP (Model Context Protocol) compatible interface for DRG.
This module provides functions that can be used by agentic coding environments.
"""

from typing import Dict, Any
from .schema import DRGSchema
from .extract import extract_typed
from .graph import KG
from .optimize import refine_triples


def build(schema: DRGSchema, text: str) -> Dict[str, Any]:
    """
    Main MCP-compatible function: Build knowledge graph from text.
    
    Args:
        schema: DRGSchema instance
        text: Input text to extract entities and relationships from
    
    Returns:
        Dictionary with 'nodes' and 'edges' keys
    """
    entities_typed, triples = extract_typed(text, schema)
    triples = refine_triples(triples)
    kg = KG.from_typed(entities_typed, triples)
    
    return {
        "nodes": [{"id": n, **attr} for n, attr in kg.nodes.items()],
        "edges": [{"source": s, "type": r, "target": o} for s, r, o in kg.edges],
    }


def build_from_file(schema: DRGSchema, filepath: str) -> Dict[str, Any]:
    """
    Build KG from a text file.
    
    Args:
        schema: DRGSchema instance
        filepath: Path to input text file
    
    Returns:
        Dictionary with 'nodes' and 'edges' keys
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return build(schema, text)
