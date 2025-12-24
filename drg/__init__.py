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
from .extract import extract_typed, extract_triples, KGExtractor, generate_schema_from_text
from .graph import KG

# Graph Module (Person 2 - Schema, KG, Visualization, Reports)
from .graph import (
    # Schema Generator
    PropertyDefinition,
    EntityClassDefinition,
    DatasetAgnosticSchemaGenerator,
    create_default_schema,
    # Relationship Model
    RelationshipType,
    EnrichedRelationship,
    RelationshipTypeClassifier,
    create_enriched_relationship,
    RELATIONSHIP_CATEGORIES,
    # KG Core
    KGNode,
    KGEdge,
    Cluster,
    EnhancedKG,
    # Visualization
    KGVisualizer,
    DEFAULT_NODE_COLORS,
    DEFAULT_EDGE_COLORS,
    # Community Report
    CommunityReport,
    CommunityReportGenerator,
)

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
    RetrievalContext,
    DRGSearch,
    create_drg_search,
    HybridRetriever,
    GraphRAGRetriever,
    create_graphrag_retriever,
    GraphRAGRetrievalContext,
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

# Optimizer
from .optimizer import (
    OptimizerConfig,
    DRGOptimizer,
    create_optimizer,
    evaluate_extraction,
    ExtractionMetrics,
    calculate_metrics,
    compare_metrics,
)

# MCP API
from .mcp_api import (
    DRGMCPAPI,
    MCPRequest,
    MCPResponse,
    MCPErrorCode,
    create_mcp_api,
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
    "generate_schema_from_text",
    # Graph (Legacy)
    "KG",
    # Graph Module (Person 2)
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
    "RetrievalContext",
    "DRGSearch",
    "create_drg_search",
    "HybridRetriever",
    "GraphRAGRetriever",
    "create_graphrag_retriever",
    "GraphRAGRetrievalContext",
    # Clustering
    "ClusteringAlgorithm",
    "LouvainClustering",
    "LeidenClustering",
    "SpectralClustering",
    "create_clustering_algorithm",
    "ClusterSummarizer",
    "create_summarizer",
    # Optimizer
    "OptimizerConfig",
    "DRGOptimizer",
    "create_optimizer",
    "evaluate_extraction",
    "ExtractionMetrics",
    "calculate_metrics",
    "compare_metrics",
    # MCP API
    "DRGMCPAPI",
    "MCPRequest",
    "MCPResponse",
    "MCPErrorCode",
    "create_mcp_api",
]
