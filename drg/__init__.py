"""DRG - Declarative Relationship Generation"""
__version__ = "0.1.0a0"

# Core schema types (lightweight - import directly)
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

# Core extraction functions (lightweight - import directly)
from .extract import extract_typed, extract_triples, KGExtractor, generate_schema_from_text

# Legacy graph class (lightweight)
from .graph import KG

# Heavy modules use lazy loading to avoid importing all dependencies at startup
# These will be imported only when explicitly requested

__all__ = [
    # Core schema classes
    "Entity",
    "Relation",
    "DRGSchema",
    "EntityType",
    "EntityGroup",
    "PropertyGroup",
    "RelationGroup",
    "EnhancedDRGSchema",
    # Core extraction
    "extract_typed",
    "extract_triples",
    "KGExtractor",
    "generate_schema_from_text",
    # Legacy graph
    "KG",
]


# Schema loading utility (available via submodule import: drg.schema.load_schema_from_json)
# Not exported at top level to keep core imports lightweight


def __getattr__(name: str):
    """Lazy loading for heavy modules that are not frequently used.
    
    This allows importing heavy dependencies only when explicitly requested,
    improving startup time and reducing memory usage.
    
    Note: Imported objects are cached in globals() for subsequent access.
    """
    # Lazy loading mapping: name -> module_path
    lazy_imports = {
        # Graph module (heavy - many submodules)
        "PropertyDefinition": ".graph",
        "EntityClassDefinition": ".graph",
        "DatasetAgnosticSchemaGenerator": ".graph",
        "create_default_schema": ".graph",
        "RelationshipType": ".graph",
        "EnrichedRelationship": ".graph",
        "RelationshipTypeClassifier": ".graph",
        "create_enriched_relationship": ".graph",
        "RELATIONSHIP_CATEGORIES": ".graph",
        "KGNode": ".graph",
        "KGEdge": ".graph",
        "Cluster": ".graph",
        "EnhancedKG": ".graph",
        "KGVisualizer": ".graph",
        "DEFAULT_NODE_COLORS": ".graph",
        "DEFAULT_EDGE_COLORS": ".graph",
        "CommunityReport": ".graph",
        "CommunityReportGenerator": ".graph",
        "Neo4jConfig": ".graph",
        "Neo4jExporter": ".graph",
        "ProvenanceNode": ".graph",
        "ProvenanceEdge": ".graph",
        "ProvenanceGraph": ".graph",
        "VisualizationAdapter": ".graph",
        # Chunking module
        "ChunkingStrategy": ".chunking",
        "TokenBasedChunker": ".chunking",
        "SentenceBasedChunker": ".chunking",
        "create_chunker": ".chunking",
        "ChunkValidator": ".chunking",
        "validate_chunks": ".chunking",
        # Embedding module
        "EmbeddingProvider": ".embedding",
        "OpenAIEmbeddingProvider": ".embedding",
        "GeminiEmbeddingProvider": ".embedding",
        "OpenRouterEmbeddingProvider": ".embedding",
        "LocalEmbeddingProvider": ".embedding",
        "create_embedding_provider": ".embedding",
        # Vector Store module
        "VectorStore": ".vector_store",
        "SearchResult": ".vector_store",
        "ChromaVectorStore": ".vector_store",
        "create_vector_store": ".vector_store",
        "QdrantVectorStore": ".vector_store",
        "PineconeVectorStore": ".vector_store",
        "FAISSVectorStore": ".vector_store",
        # Retrieval module
        "RAGRetriever": ".retrieval",
        "create_rag_retriever": ".retrieval",
        "RetrievalContext": ".retrieval",
        "DRGSearch": ".retrieval",
        "create_drg_search": ".retrieval",
        "HybridRetriever": ".retrieval",
        "GraphRAGRetriever": ".retrieval",
        "create_graphrag_retriever": ".retrieval",
        "GraphRAGRetrievalContext": ".retrieval",
        # Clustering module
        "ClusteringAlgorithm": ".clustering",
        "LouvainClustering": ".clustering",
        "LeidenClustering": ".clustering",
        "SpectralClustering": ".clustering",
        "create_clustering_algorithm": ".clustering",
        "ClusterSummarizer": ".clustering",
        "create_summarizer": ".clustering",
        # Optimizer module (heavy - requires DSPy)
        "OptimizerConfig": ".optimizer",
        "DRGOptimizer": ".optimizer",
        "create_optimizer": ".optimizer",
        "evaluate_extraction": ".optimizer",
        "ExtractionMetrics": ".optimizer",
        "calculate_metrics": ".optimizer",
        "compare_metrics": ".optimizer",
        # MCP API module (heavy)
        "DRGMCPAPI": ".mcp_api",
        "MCPRequest": ".mcp_api",
        "MCPResponse": ".mcp_api",
        "MCPErrorCode": ".mcp_api",
        "create_mcp_api": ".mcp_api",
    }
    
    if name in lazy_imports:
        module_path = lazy_imports[name]
        # Import the module
        import importlib
        module = importlib.import_module(module_path, __name__)
        # Get the requested attribute
        attr = getattr(module, name)
        # Cache it in globals for subsequent access
        globals()[name] = attr
        return attr
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
