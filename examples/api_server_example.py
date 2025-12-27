"""
DRG API Server Example

This example demonstrates how to:
1. Load a knowledge graph
2. Create a GraphRAG retriever
3. Start the FastAPI server
4. Access the web UI and API endpoints

Usage:
    # Set API key as environment variable (Gemini - önerilen)
    export GEMINI_API_KEY=AIzaSy...
    python examples/api_server_example.py
    
    # Or use OpenAI (alternatif)
    export OPENAI_API_KEY=sk-or-v1-...
    python examples/api_server_example.py
    
    # Or set Neo4j credentials (optional)
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg.graph import EnhancedKG
from drg.graph.neo4j_exporter import Neo4jConfig
from drg.graph.kg_core import KGNode, KGEdge, Cluster
from drg.embedding.providers import create_embedding_provider
from drg.retrieval.graphrag import GraphRAGRetriever
from drg.api import DRGAPIServer


def create_sample_kg() -> EnhancedKG:
    """Create a sample knowledge graph for demonstration."""
    kg = EnhancedKG()
    
    # Add sample nodes
    nodes = [
        KGNode(id="Apple", type="Company", properties={"industry": "Technology"}),
        KGNode(id="iPhone", type="Product", properties={"category": "Smartphone"}),
        KGNode(id="Tim Cook", type="Person", properties={"role": "CEO"}),
        KGNode(id="Cupertino", type="Location", properties={"type": "City"}),
        KGNode(id="Google", type="Company", properties={"industry": "Technology"}),
        KGNode(id="Android", type="Product", properties={"category": "OS"}),
    ]
    
    for node in nodes:
        kg.add_node(node)
    
    # Add sample edges
    edges = [
        KGEdge(
            source="Apple",
            target="iPhone",
            relationship_type="produces",
            relationship_detail="Apple manufactures iPhone",
        ),
        KGEdge(
            source="Tim Cook",
            target="Apple",
            relationship_type="works_for",
            relationship_detail="Tim Cook is CEO of Apple",
        ),
        KGEdge(
            source="Apple",
            target="Cupertino",
            relationship_type="located_in",
            relationship_detail="Apple headquarters is in Cupertino",
        ),
        KGEdge(
            source="Google",
            target="Android",
            relationship_type="produces",
            relationship_detail="Google develops Android",
        ),
    ]
    
    for edge in edges:
        kg.add_edge(edge)
    
    # Add sample cluster
    cluster = Cluster(
        id="tech_companies",
        node_ids={"Apple", "Google", "iPhone", "Android"},
        metadata={"theme": "Technology companies and products"},
    )
    kg.add_cluster(cluster)
    
    return kg


def main():
    """Main function to start the API server."""
    import sys
    import glob
    from datetime import datetime
    
    # Example name: command line argument > environment variable > auto-detect latest > default
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name.isdigit():
            example_name = f"{example_name}example"
    elif os.getenv("DRG_EXAMPLE"):
        example_name = os.getenv("DRG_EXAMPLE")
        if example_name.isdigit():
            example_name = f"{example_name}example"
    else:
        # Auto-detect: Find the most recently modified KG file
        kg_files = list(Path("outputs").glob("*example*_kg.json"))
        if kg_files:
            # Sort by modification time (most recent first)
            latest_kg = max(kg_files, key=lambda p: p.stat().st_mtime)
            # Extract example name from filename (e.g., "outputs/3example_kg.json" -> "3example")
            example_name = latest_kg.stem.replace("_kg", "")
            print(f"🔍 En son güncellenen KG dosyası bulundu: {latest_kg.name}")
        else:
            example_name = "1example"
    
    print("=" * 70)
    print(f"DRG API Server Example - {example_name.upper()}")
    print("=" * 70)
    print(f"📌 Example seçimi: {example_name}")
    print(f"   (Değiştirmek için: export DRG_EXAMPLE=3example veya python {sys.argv[0]} 3)")
    
    # Load knowledge graph from file if exists, otherwise use sample
    kg_path = Path(f"outputs/{example_name}_kg.json")
    if kg_path.exists():
        print("\n1. Loading knowledge graph from file...")
        import json
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
        
        kg = EnhancedKG()
        # Load nodes
        for node_data in kg_data.get("nodes", []):
            node = KGNode.from_dict(node_data)
            kg.add_node(node)
        # Load edges
        for edge_data in kg_data.get("edges", []):
            edge = KGEdge.from_dict(edge_data)
            # Make sure source and target nodes exist
            if edge.source not in kg.nodes:
                kg.add_node(KGNode(id=edge.source, type=None))
            if edge.target not in kg.nodes:
                kg.add_node(KGNode(id=edge.target, type=None))
            kg.add_edge(edge)
        # Load clusters
        for cluster_data in kg_data.get("clusters", []):
            cluster = Cluster.from_dict(cluster_data)
            kg.add_cluster(cluster)
        print(f"   ✅ Loaded KG with {len(kg.nodes)} nodes, {len(kg.edges)} edges, {len(kg.clusters)} clusters")
    else:
        print("\n1. Creating sample knowledge graph...")
        kg = create_sample_kg()
        print(f"   ✅ Created KG with {len(kg.nodes)} nodes, {len(kg.edges)} edges, {len(kg.clusters)} clusters")
    
    # Create embedding provider (optional - only needed for retrieval)
    print("\n2. Creating embedding provider...")
    retriever = None
    
    # Check for API key (try Gemini first, then OpenAI)
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        print("   ⚠️  GEMINI_API_KEY or OPENAI_API_KEY environment variable not set")
        print("   ℹ️  Set it with: export GEMINI_API_KEY=AIzaSy... or export OPENAI_API_KEY=sk-...")
        print("   ℹ️  API server will work but query functionality will be limited")
    else:
        try:
            # Try Gemini first if available, otherwise OpenAI
            if gemini_key:
                print("   ℹ️  Using Gemini embedding provider")
                embedding_provider = create_embedding_provider(
                    provider="gemini",
                    model="models/embedding-001",
                )
            else:
                print("   ℹ️  Using OpenAI embedding provider")
                embedding_provider = create_embedding_provider(
                    provider="openai",
                    model="text-embedding-3-small",
                )
            print("   ✅ Embedding provider created")
            
            # Add embeddings to entities
            kg.add_entity_embeddings(embedding_provider)
            print("   ✅ Entity embeddings added")
            
            # Create GraphRAG retriever
            retriever = GraphRAGRetriever(
                embedding_provider=embedding_provider,
                knowledge_graph=kg,
                similarity_threshold=0.3,
                max_hops=2,
            )
            print("   ✅ GraphRAG retriever created")
        except Exception as e:
            print(f"   ⚠️  Could not create embedding provider/retriever: {e}")
            print("   ℹ️  API server will work but query functionality will be limited")
            retriever = None
    
    # Optional: Neo4j configuration (comment out if not using Neo4j)
    neo4j_config = None
    neo4j_uri = os.getenv("NEO4J_URI")
    if neo4j_uri:
        neo4j_config = Neo4jConfig(
            uri=neo4j_uri,
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        print("\n3. Neo4j configuration loaded")
    else:
        print("\n3. Neo4j not configured (set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD to enable)")
    
    # Create API server
    print("\n4. Creating API server...")
    server = DRGAPIServer(
        kg=kg,
        neo4j_config=neo4j_config,
        graphrag_retriever=retriever,
    )
    print("   ✅ API server created")
    
    # Start server
    print("\n" + "=" * 70)
    print("Starting DRG API Server...")
    print("=" * 70)
    print("\n🌐 Web UI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Graph API: http://localhost:8000/api/graph")
    print("👥 Communities API: http://localhost:8000/api/communities")
    print("\nPress Ctrl+C to stop the server\n")
    
    server.run(host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
