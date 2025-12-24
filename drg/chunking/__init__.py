"""Chunking module for dataset-agnostic text segmentation."""

from .strategies import (
    ChunkingStrategy,
    TokenBasedChunker,
    SentenceBasedChunker,
    create_chunker,
    CHUNKING_PRESETS,
)
from .validators import ChunkValidator, validate_chunks

__all__ = [
    "ChunkingStrategy",
    "TokenBasedChunker",
    "SentenceBasedChunker",
    "create_chunker",
    "CHUNKING_PRESETS",
    "ChunkValidator",
    "validate_chunks",
]

