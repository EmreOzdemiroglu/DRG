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

# Chunking
from .chunking import (
    ChunkingStrategy,
    TokenBasedChunker,
    SentenceBasedChunker,
    create_chunker,
    ChunkValidator,
    validate_chunks,
)

# Embedding
from .embedding import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    LocalEmbeddingProvider,
    create_embedding_provider,
)

# Vector Store
from .vector_store import (
    VectorStore,
    SearchResult,
    ChromaVectorStore,
    create_vector_store,
)

# Retrieval
from .retrieval import (
    RAGRetriever,
    create_rag_retriever,
    DRGSearch,
    create_drg_search,
    HybridRetriever,
)

# Clustering
from .clustering import (
    ClusteringAlgorithm,
    LouvainClustering,
    LeidenClustering,
    SpectralClustering,
    create_clustering_algorithm,
    ClusterSummarizer,
    create_summarizer,
)

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
    # Chunking
    "ChunkingStrategy",
    "TokenBasedChunker",
    "SentenceBasedChunker",
    "create_chunker",
    "ChunkValidator",
    "validate_chunks",
    # Embedding
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "OpenRouterEmbeddingProvider",
    "LocalEmbeddingProvider",
    "create_embedding_provider",
    # Vector Store
    "VectorStore",
    "SearchResult",
    "ChromaVectorStore",
    "create_vector_store",
    # Retrieval
    "RAGRetriever",
    "create_rag_retriever",
    "DRGSearch",
    "create_drg_search",
    "HybridRetriever",
    # Clustering
    "ClusteringAlgorithm",
    "LouvainClustering",
    "LeidenClustering",
    "SpectralClustering",
    "create_clustering_algorithm",
    "ClusterSummarizer",
    "create_summarizer",
]
