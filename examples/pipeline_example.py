"""Example: Complete pipeline with chunking, embedding, and RAG retrieval."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drg.chunking import create_chunker, TokenBasedChunker
from drg.embedding import create_embedding_provider
from drg.vector_store import create_vector_store
from drg.retrieval import create_rag_retriever
from drg.graph import KG
from drg.extract import extract_typed
from drg.schema import DRGSchema, Entity, Relation


def main():
    """Run complete pipeline example."""
    
    # Set Gemini as default model for extraction
    import os
    if not os.getenv("DRG_MODEL"):
        os.environ["DRG_MODEL"] = "gemini/gemini-2.0-flash-exp"
    
    # Sample text
    text = """
    Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
    The company produces various products including the iPhone, iPad, and Mac computers.
    Tim Cook is the current CEO of Apple.
    Apple's headquarters is located in Cupertino, California.
    The iPhone was first released in 2007 and revolutionized the smartphone industry.
    """
    
    print("=" * 60)
    print("DRG Pipeline Example: Chunking + Embedding + RAG")
    print("=" * 60)
    
    # Step 1: Chunking
    print("\n1. Chunking text...")
    chunker = create_chunker(
        strategy="token_based",
        chunk_size=100,  # Small for demo
        overlap_ratio=0.15,
    )
    
    chunks = chunker.chunk(
        text=text,
        origin_dataset="demo",
        origin_file="example.txt",
    )
    
    print(f"   Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i}: {chunk.token_count} tokens, ID: {chunk.chunk_id}")
    
    # Step 2: Knowledge Graph Extraction
    print("\n2. Extracting knowledge graph...")
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Person"), Entity("Product"), Entity("Location")],
        relations=[
            Relation("founded_by", "Company", "Person"),
            Relation("produces", "Company", "Product"),
            Relation("located_in", "Company", "Location"),
            Relation("ceo_of", "Person", "Company"),
        ],
    )
    
    # Try extraction, fallback to mock data if API fails
    try:
        entities, triples = extract_typed(text, schema)
        kg = KG.from_typed(entities, triples)
        print(f"   Extracted {len(entities)} entities and {len(triples)} relations")
    except Exception as e:
        print(f"   KG extraction failed ({type(e).__name__}), using mock data...")
        # Mock KG for demo
        entities = [("Apple", "Company"), ("Steve Jobs", "Person"), ("iPhone", "Product")]
        triples = [("Apple", "produces", "iPhone"), ("Steve Jobs", "founded_by", "Apple")]
        kg = KG.from_typed(entities, triples)
        print(f"   Using mock: {len(entities)} entities and {len(triples)} relations")
    
    # Step 3: Embedding
    print("\n3. Creating embeddings...")
    
    # Try Gemini first, fallback to local, then mock
    embedding_provider = None
    try:
        embedding_provider = create_embedding_provider(
            provider="gemini",
            model="models/embedding-001",
        )
        # Test with a small text first
        _ = embedding_provider.embed("test")
        print("   Using Gemini embeddings")
    except Exception as e:
        print(f"   Gemini not available ({type(e).__name__}), trying local...")
        try:
            embedding_provider = create_embedding_provider(
                provider="local",
                model="sentence-transformers/all-MiniLM-L6-v2",
            )
            print("   Using local embeddings")
        except Exception as e2:
            print(f"   Local embeddings not available ({type(e2).__name__}), using mock embeddings...")
            # Mock embedding provider for demo
            class MockEmbeddingProvider:
                def embed(self, text: str):
                    import random
                    return [random.random() for _ in range(384)]
                def embed_batch(self, texts):
                    return [self.embed(text) for text in texts]
                def get_dimension(self):
                    return 384
                def get_model_name(self):
                    return "mock/random"
            embedding_provider = MockEmbeddingProvider()
            print("   Using mock embeddings (random vectors)")
    
    # Embed chunks
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_embeddings = embedding_provider.embed_batch(chunk_texts)
    print(f"   Created {len(chunk_embeddings)} embeddings (dim: {embedding_provider.get_dimension()})")
    
    # Step 4: Vector Store
    print("\n4. Storing in vector database...")
    try:
        vector_store = create_vector_store(
            store_type="chroma",
            collection_name="demo_chunks",
        )
        print("   Using Chroma vector store")
    except Exception as e:
        print(f"   Chroma not available ({type(e).__name__}), using mock vector store...")
        # Mock vector store for demo
        from drg.vector_store.interface import VectorStore, SearchResult
        class MockVectorStore(VectorStore):
            def __init__(self):
                self.data = {}
            def add(self, embeddings, metadata, ids=None):
                if ids is None:
                    ids = [f"chunk_{i}" for i in range(len(embeddings))]
                for i, (emb, meta, id) in enumerate(zip(embeddings, metadata, ids)):
                    self.data[id] = {"embedding": emb, "metadata": meta}
                return ids
            def search(self, query_embedding, k=10, filters=None):
                # Simple cosine similarity search
                results = []
                for id, item in self.data.items():
                    emb = item["embedding"]
                    # Cosine similarity
                    dot = sum(a*b for a,b in zip(query_embedding, emb))
                    mag1 = sum(a*a for a in query_embedding)**0.5
                    mag2 = sum(b*b for b in emb)**0.5
                    score = dot / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0
                    results.append(SearchResult(
                        chunk_id=id,
                        score=score,
                        metadata=item["metadata"]
                    ))
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:k]
            def update(self, id, embedding=None, metadata=None):
                if id in self.data:
                    if embedding: self.data[id]["embedding"] = embedding
                    if metadata: self.data[id]["metadata"] = metadata
            def delete(self, ids):
                for id in ids:
                    self.data.pop(id, None)
            def get_metadata(self, id):
                return self.data.get(id, {}).get("metadata")
            def count(self):
                return len(self.data)
        vector_store = MockVectorStore()
        print("   Using mock vector store (in-memory)")
    
    # Prepare metadata
    chunk_metadata = [chunk.to_dict() for chunk in chunks]
    chunk_ids = vector_store.add(
        embeddings=chunk_embeddings,
        metadata=chunk_metadata,
    )
    print(f"   Stored {len(chunk_ids)} chunks in vector store")
    
    # Step 5: RAG Retrieval with KG Context
    print("\n5. RAG Retrieval with Knowledge Graph context...")
    rag_retriever = create_rag_retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        knowledge_graph=kg,
        include_kg_context=True,
    )
    
    # Test query
    query = "What products does Apple produce?"
    print(f"\n   Query: {query}")
    
    context = rag_retriever.retrieve(query=query, k=3)
    
    print(f"\n   Retrieved {len(context.chunks)} chunks:")
    for i, chunk in enumerate(context.chunks, 1):
        print(f"   {i}. Score: {chunk['score']:.3f}")
        print(f"      Text: {chunk['text'][:100]}...")
    
    if context.kg_subgraph:
        print(f"\n   Knowledge Graph Context:")
        print(f"   - Entities: {len(context.entities)}")
        print(f"   - Relationships: {len(context.relationships)}")
        if context.entities:
            print(f"   - Entity list: {', '.join(context.entities[:5])}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

