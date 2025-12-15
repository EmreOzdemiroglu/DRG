"""Embedding abstraction layer for semantic representations."""

from .providers import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    LocalEmbeddingProvider,
    create_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "OpenRouterEmbeddingProvider",
    "LocalEmbeddingProvider",
    "create_embedding_provider",
]

