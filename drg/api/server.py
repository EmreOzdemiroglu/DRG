"""
FastAPI Server for DRG Knowledge Graph API

Endpoints:
- GET /api/graph - Full graph data
- GET /api/graph/stats - Graph statistics
- GET /api/communities - Community/cluster data
- GET /api/communities/{cluster_id} - Specific community report
- GET /api/provenance/{query_id} - Query provenance chain
- POST /api/query - Execute query and get provenance
- GET /api/visualization/{format} - Graph visualization data (cytoscape, vis-network, d3)
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    FastAPI = None

from ..graph import (
    EnhancedKG,
    Neo4jExporter,
    Neo4jConfig,
    VisualizationAdapter,
    ProvenanceGraph,
)
from ..retrieval.graphrag import GraphRAGRetriever, GraphRAGRetrievalContext

logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    k_entities: int = 10
    k_reports: int = 5
    k_context_chunks: int = 5


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: Optional[str] = None
    provenance_id: str
    retrieval_context: Dict[str, Any]


def create_app(
    kg: Optional[EnhancedKG] = None,
    neo4j_config: Optional[Neo4jConfig] = None,
    graphrag_retriever: Optional[GraphRAGRetriever] = None,
    provenance_store: Optional[Dict[str, ProvenanceGraph]] = None,
) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        kg: EnhancedKG instance
        neo4j_config: Optional Neo4j configuration for persistence
        graphrag_retriever: Optional GraphRAG retriever for queries
        provenance_store: Optional in-memory provenance store
    
    Returns:
        FastAPI application instance
    """
    if FastAPI is None:
        raise ImportError(
            "fastapi and uvicorn are required. Install with: pip install fastapi uvicorn"
        )
    
    app = FastAPI(
        title="DRG Knowledge Graph API",
        description="API for DRG Knowledge Graph visualization and querying",
        version="1.0.0",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store app state
    app.state.kg = kg
    app.state.neo4j_config = neo4j_config
    app.state.graphrag_retriever = graphrag_retriever
    app.state.provenance_store = provenance_store or {}
    app.state.visualization_adapter = VisualizationAdapter(kg) if kg else None
    
    # Mount static files (for web UI)
    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint - serve web UI."""
        ui_file = templates_dir / "index.html"
        if ui_file.exists():
            return ui_file.read_text(encoding="utf-8")
        return "<h1>DRG Knowledge Graph API</h1><p>Use /docs for API documentation</p>"
    
    @app.get("/api/graph")
    async def get_graph():
        """Get full graph data."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        return {
            "nodes": [node.to_dict() for node in kg.nodes.values()],
            "edges": [edge.to_dict() for edge in kg.edges],
            "clusters": [cluster.to_dict() for cluster in kg.clusters.values()],
        }
    
    @app.get("/api/graph/stats")
    async def get_graph_stats():
        """Get graph statistics."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        # Calculate statistics
        node_types = {}
        for node in kg.nodes.values():
            node_type = node.type or "Unknown"
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        relationship_types = {}
        for edge in kg.edges:
            rel_type = edge.relationship_type
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "node_count": len(kg.nodes),
            "edge_count": len(kg.edges),
            "cluster_count": len(kg.clusters),
            "node_types": node_types,
            "relationship_types": relationship_types,
        }
    
    @app.get("/api/communities")
    async def get_communities():
        """Get all communities/clusters."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        return {
            "clusters": [cluster.to_dict() for cluster in kg.clusters.values()],
        }
    
    @app.get("/api/communities/{cluster_id}")
    async def get_community(cluster_id: str):
        """Get specific community report."""
        kg = app.state.kg
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        cluster = kg.clusters.get(cluster_id)
        if cluster is None:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
        
        # Generate community report if generator available
        from ..graph.community_report import CommunityReportGenerator
        
        report_generator = CommunityReportGenerator(kg)
        report = report_generator.generate_report(cluster)
        
        return report.to_dict()
    
    @app.get("/api/visualization/{format}")
    async def get_visualization(format: str):
        """Get graph visualization data in specified format."""
        kg = app.state.kg
        adapter = app.state.visualization_adapter
        
        if kg is None or adapter is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        format_lower = format.lower()
        
        if format_lower == "cytoscape":
            data = adapter.kg_to_cytoscape(kg)
            return {"elements": data}
        
        elif format_lower == "vis-network" or format_lower == "visnetwork":
            data = adapter.kg_to_vis_network(kg)
            return data
        
        elif format_lower == "d3":
            data = adapter.kg_to_d3_json(kg)
            return data
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported: cytoscape, vis-network, d3"
            )
    
    @app.get("/api/visualization/communities/{format}")
    async def get_communities_visualization(format: str):
        """Get communities visualization with color coding."""
        kg = app.state.kg
        adapter = app.state.visualization_adapter
        
        if kg is None or adapter is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        format_lower = format.lower()
        
        if format_lower == "cytoscape":
            data = adapter.communities_to_cytoscape(kg)
            return {"elements": data}
        
        else:
            # For other formats, use regular visualization
            return await get_visualization(format)
    
    @app.post("/api/query")
    async def execute_query(request: QueryRequest):
        """Execute query and get provenance chain."""
        retriever = app.state.graphrag_retriever
        if retriever is None:
            raise HTTPException(
                status_code=404,
                detail="GraphRAG retriever not configured"
            )
        
        # Execute retrieval
        retrieval_context = retriever.retrieve(
            query=request.query,
            k_entities=request.k_entities,
            k_reports=request.k_reports,
            k_context_chunks=request.k_context_chunks,
        )
        
        # Generate provenance graph (simplified - in production, would use LLM to generate answer)
        provenance = _create_provenance_from_retrieval(request.query, retrieval_context)
        
        # Store provenance
        provenance_id = f"provenance_{len(app.state.provenance_store)}"
        app.state.provenance_store[provenance_id] = provenance
        
        return {
            "query": request.query,
            "answer": provenance.answer,
            "provenance_id": provenance_id,
            "retrieval_context": {
                "seed_entities": retrieval_context.seed_entities,
                "entities": retrieval_context.entities,
                "relationships": retrieval_context.relationships,
                "community_reports": retrieval_context.community_reports,
            },
        }
    
    @app.get("/api/provenance/{provenance_id}")
    async def get_provenance(
        provenance_id: str,
        format: str = "json"
    ):
        """Get query provenance chain."""
        provenance_store = app.state.provenance_store
        adapter = app.state.visualization_adapter
        
        if provenance_id not in provenance_store:
            raise HTTPException(
                status_code=404,
                detail=f"Provenance {provenance_id} not found"
            )
        
        provenance = provenance_store[provenance_id]
        
        format_lower = format.lower()
        
        if format_lower == "cytoscape" and adapter:
            data = adapter.provenance_to_cytoscape(provenance)
            return {"elements": data}
        
        else:
            if adapter:
                return adapter.provenance_to_json(provenance)
            else:
                return {
                    "query": provenance.query,
                    "answer": provenance.answer,
                    "nodes": [
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            "data": node.data,
                        }
                        for node in provenance.nodes
                    ],
                    "edges": [
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.type,
                            "label": edge.label,
                        }
                        for edge in provenance.edges
                    ],
                }
    
    @app.post("/api/neo4j/sync")
    async def sync_to_neo4j(clear_existing: bool = Query(False)):
        """Sync knowledge graph to Neo4j."""
        kg = app.state.kg
        neo4j_config = app.state.neo4j_config
        
        if kg is None:
            raise HTTPException(status_code=404, detail="Knowledge graph not loaded")
        
        if neo4j_config is None:
            raise HTTPException(
                status_code=404,
                detail="Neo4j configuration not provided"
            )
        
        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.sync_kg(kg, clear_existing=clear_existing)
            exporter.close()
            
            return {
                "status": "success",
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Neo4j sync failed: {str(e)}")
    
    @app.get("/api/neo4j/stats")
    async def get_neo4j_stats():
        """Get Neo4j graph statistics."""
        neo4j_config = app.state.neo4j_config
        
        if neo4j_config is None:
            raise HTTPException(
                status_code=404,
                detail="Neo4j configuration not provided"
            )
        
        try:
            exporter = Neo4jExporter(neo4j_config)
            stats = exporter.get_graph_stats()
            exporter.close()
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get Neo4j stats: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get Neo4j stats: {str(e)}")
    
    return app


def _create_provenance_from_retrieval(
    query: str,
    retrieval_context: GraphRAGRetrievalContext,
) -> ProvenanceGraph:
    """Create provenance graph from retrieval context.
    
    Creates provenance chain: query → chunks → community → summary → answer
    """
    from ..graph.visualization_adapter import ProvenanceNode, ProvenanceEdge
    
    nodes = []
    edges = []
    
    # Query node
    query_node = ProvenanceNode(
        id="query",
        type="query",
        label=f"Query: {query[:50]}",
        data={"query": query},
    )
    nodes.append(query_node)
    
    # Chunk nodes
    chunk_nodes = []
    if retrieval_context.context_chunks:
        for idx, chunk in enumerate(retrieval_context.context_chunks[:5]):  # Limit to 5 chunks
            chunk_id = f"chunk_{idx}"
            chunk_node = ProvenanceNode(
                id=chunk_id,
                type="chunk",
                label=f"Chunk {idx+1}",
                data={
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text", "")[:100],
                    "score": chunk.get("score"),
                },
                metadata=chunk.get("metadata", {}),
            )
            nodes.append(chunk_node)
            chunk_nodes.append(chunk_id)
            
            # Edge from query to chunk
            edges.append(ProvenanceEdge(
                source="query",
                target=chunk_id,
                type="retrieved_from",
                label="retrieved",
                weight=chunk.get("score", 1.0),
            ))
    
    # Community nodes
    community_nodes = []
    for idx, report in enumerate(retrieval_context.community_reports[:5]):  # Limit to 5 communities
        community_id = f"community_{report.get('cluster_id', idx)}"
        community_node = ProvenanceNode(
            id=community_id,
            type="community",
            label=f"Community {idx+1}",
            data={
                "cluster_id": report.get("cluster_id"),
                "summary": report.get("summary", "")[:100],
            },
            metadata=report.get("metadata", {}),
        )
        nodes.append(community_node)
        community_nodes.append(community_id)
        
        # Edge from query to community (via seed entities)
        edges.append(ProvenanceEdge(
            source="query",
            target=community_id,
            type="matched_community",
            label="matched",
            weight=1.0,
        ))
        
        # Summary node
        summary_id = f"summary_{idx}"
        summary_node = ProvenanceNode(
            id=summary_id,
            type="summary",
            label=f"Summary {idx+1}",
            data={
                "summary": report.get("summary", "")[:200],
                "themes": report.get("themes", []),
            },
            metadata=report.get("metadata", {}),
        )
        nodes.append(summary_node)
        
        # Edge from community to summary
        edges.append(ProvenanceEdge(
            source=community_id,
            target=summary_id,
            type="summarized_in",
            label="summarized",
            weight=1.0,
        ))
        
        # Edge from summary to answer
        if idx == 0:  # Connect first summary to answer
            answer_id = "answer"
            answer_node = ProvenanceNode(
                id=answer_id,
                type="answer",
                label="Answer",
                data={
                    "answer": f"Based on {len(retrieval_context.community_reports)} communities and {len(retrieval_context.seed_entities)} entities",
                },
            )
            nodes.append(answer_node)
            
            edges.append(ProvenanceEdge(
                source=summary_id,
                target=answer_id,
                type="generated_from",
                label="generated",
                weight=1.0,
            ))
    
    # Generate answer text
    answer_text = f"Query: {query}\n\n"
    answer_text += f"Found {len(retrieval_context.seed_entities)} relevant entities in "
    answer_text += f"{len(retrieval_context.community_reports)} communities.\n\n"
    
    if retrieval_context.community_reports:
        answer_text += "Community summaries:\n"
        for report in retrieval_context.community_reports[:3]:
            answer_text += f"- {report.get('summary', '')}\n"
    
    provenance = ProvenanceGraph(
        nodes=nodes,
        edges=edges,
        query=query,
        answer=answer_text,
        metadata={
            "seed_entities": retrieval_context.seed_entities,
            "entity_count": len(retrieval_context.entities),
            "relationship_count": len(retrieval_context.relationships),
        },
    )
    
    return provenance


class DRGAPIServer:
    """DRG API Server wrapper class."""
    
    def __init__(
        self,
        kg: Optional[EnhancedKG] = None,
        neo4j_config: Optional[Neo4jConfig] = None,
        graphrag_retriever: Optional[GraphRAGRetriever] = None,
    ):
        """Initialize API server.
        
        Args:
            kg: EnhancedKG instance
            neo4j_config: Optional Neo4j configuration
            graphrag_retriever: Optional GraphRAG retriever
        """
        self.kg = kg
        self.neo4j_config = neo4j_config
        self.graphrag_retriever = graphrag_retriever
        self.app = create_app(
            kg=kg,
            neo4j_config=neo4j_config,
            graphrag_retriever=graphrag_retriever,
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload (development)
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required. Install with: pip install uvicorn"
            )
        
        uvicorn.run(self.app, host=host, port=port, reload=reload)

