# DRG Arama Algoritmaları: GraphRAG Erişim Tasarımı

## 1. Genel Bakış

DRG (Dynamic Retrieval Graph) search, knowledge graph structure'ını kullanarak semantic retrieval yapar. Classic RAG'ın aksine, sadece vector similarity'ye değil, graph topology'ye de dayanır.

## 2. DRG vs Classic RAG

### 2.1 Classic RAG Limitations

**Vector Search Only:**
- Sadece semantic similarity'ye dayalı
- Graph structure ignore edilir
- Entity relationships kullanılmaz
- Multi-hop reasoning yok

**Örnek Senaryo:**

Query: "What products does Apple produce?"

Classic RAG:
1. Query'yi embed et
2. Vector similarity search yap
3. "Apple produces iPhone" chunk'ını bulur
4. Ama "Apple → produces → iPhone" relationship'ini explicit olarak kullanmaz

### 2.2 DRG Advantages

**Graph-Aware Retrieval:**
- Entity relationships explicit olarak kullanılır
- Multi-hop reasoning mümkün
- Graph topology'den bilgi çıkarılır
- Entity-centric queries için daha iyi

**Örnek Senaryo:**

Query: "What products does Apple produce?"

DRG Search:
1. Query'den "Apple" entity'sini extract et
2. Graph'da "Apple" node'unu bul
3. "produces" relation'larını traverse et
4. "iPhone", "iPad", "Mac" node'larını bul
5. Bu node'ları içeren chunk'ları retrieve et

## 3. Graph Construction

### 3.1 Graph Representation

**Node Structure:**

```
Node {
  id: "entity_name",
  type: "EntityType",
  embedding: [0.1, 0.2, ...],
  source_chunks: ["chunk_id_1", "chunk_id_2"],
  metadata: {...}
}
```

**Edge Structure:**

```
Edge {
  source: "entity_1",
  target: "entity_2",
  relation: "relation_name",
  weight: 1.0,
  source_chunks: ["chunk_id_1"],
  metadata: {...}
}
```

### 3.2 Graph Storage

**Options:**
- **Neo4j**: Production-ready graph database
- **NetworkX**: In-memory graph, Python-native
- **ArangoDB**: Multi-model database (document + graph)

**Selection Criteria:**
- **Scale**: Small (< 10K nodes) → NetworkX, Large (> 10K nodes) → Neo4j/ArangoDB
- **Performance**: Real-time → Neo4j, Batch → NetworkX
- **Features**: Advanced queries → Neo4j, Simple traversal → NetworkX

## 4. DRG Search Algorithms

### 4.1 Breadth-First Search (BFS)

**Algoritma:**

1. Query'den seed entity'leri extract et
2. Seed entity'leri queue'ya ekle
3. BFS ile graph'ı traverse et
4. Her node için semantic similarity hesapla
5. Top-K node'ları seç
6. Node'ları içeren chunk'ları retrieve et

**Scoring:**

```
node_score = α * semantic_similarity(query_embedding, node_embedding) 
           + (1 - α) * graph_distance_penalty(seed, node)
```

**graph_distance_penalty:**

```
penalty = 1 / (1 + hop_count)
```

**Avantajlar:**
- Basit, anlaşılır
- Shortest path garantisi
- Deterministic

**Dezavantajlar:**
- Tüm komşuları explore eder (inefficient)
- Semantic similarity'yi secondary olarak kullanır

### 4.2 Depth-First Search (DFS)

**Algoritma:**

1. Query'den seed entity'leri extract et
2. Seed entity'lerden DFS başlat
3. Her node'da semantic similarity kontrol et
4. Eğer similarity threshold'u geçerse, o branch'i explore et
5. Top-K node'ları seç
6. Node'ları içeren chunk'ları retrieve et

**Scoring:**

```
node_score = semantic_similarity(query_embedding, node_embedding)
           - depth_penalty * current_depth
```

**depth_penalty:**

```
penalty = 0.1 * depth  (her hop için 0.1 penalty)
```

**Avantajlar:**
- Deep exploration
- Semantic-guided traversal
- Efficient (early stopping)

**Dezavantajlar:**
- Local optima riski
- Non-deterministic (semantic similarity'ye bağlı)

### 4.3 Weighted/Hybrid Search

**Algoritma:**

1. Query'den seed entity'leri extract et
2. Seed entity'lerden weighted traversal başlat
3. Her node için hybrid score hesapla:
   - Semantic similarity
   - Graph distance
   - Edge weights
   - Node degree (importance)
4. Priority queue ile top-K node'ları seç
5. Node'ları içeren chunk'ları retrieve et

**Scoring:**

```
node_score = α * semantic_similarity(query_embedding, node_embedding)
           + β * graph_proximity_score(seed, node)
           + γ * edge_weight_score(path)
           + δ * node_importance_score(node)
```

**graph_proximity_score:**

```
proximity = 1 / (1 + shortest_path_length(seed, node))
```

**edge_weight_score:**

```
weight_score = product(edge_weights along path)
```

**node_importance_score:**

```
importance = log(1 + node_degree) / log(max_degree)
```

**Avantajlar:**
- Balanced approach
- Multiple signals kullanır
- Tunable parameters (α, β, γ, δ)

**Dezavantajlar:**
- Parameter tuning gerekli
- Computational cost yüksek

### 4.4 Multi-Hop Reasoning

**Algoritma:**

1. Query'den seed entity'leri extract et
2. Multi-hop paths bul (2-3 hop)
3. Her path için semantic coherence hesapla
4. Top-K paths seç
5. Path'teki node'ları içeren chunk'ları retrieve et

**Path Scoring:**

```
path_score = semantic_coherence(path_nodes, query)
           * path_length_penalty(path_length)
```

**semantic_coherence:**

```
coherence = average(semantic_similarity(query, node) for node in path)
```

**Avantajlar:**
- Complex queries için uygun
- Relationship chains'i capture eder

**Dezavantajlar:**
- Computational cost çok yüksek
- Path explosion riski

## 5. Hybrid RAG + GraphRAG

### 5.1 Two-Stage Retrieval

**Stage 1: Vector Search (RAG)**
- Query'yi embed et
- Vector similarity search, top-100 chunk'ları bul

**Stage 2: Graph Search (DRG)**
- Top-100 chunk'lardan entity'leri extract et
- Graph'da bu entity'lerden traverse et
- Additional chunk'ları bul

**Fusion:**

```
final_chunks = rank_fusion(rag_chunks, drg_chunks)
```

### 5.2 Reciprocal Rank Fusion

**Algoritma:**

1. RAG ve DRG sonuçlarını al
2. Her chunk için rank-based score hesapla
3. İki score'u birleştir

**Scoring:**

```
rag_score = 1 / (k + rag_rank)
drg_score = 1 / (k + drg_rank)
final_score = rag_score + drg_score
```

**k parameter:**
- Genellikle 60
- Rank sensitivity'yi kontrol eder

## 6. When DRG Outperforms RAG

### 6.1 Entity-Centric Queries

**Örnek:**
- "What companies does Elon Musk own?"
- "Who are the competitors of Apple?"

**Neden DRG Daha İyi:**
- Entity relationships explicit
- Multi-hop traversal (Musk → Tesla → competitors)
- Graph structure'ı kullanır

### 6.2 Relational Queries

**Örnek:**
- "What is the relationship between X and Y?"
- "How are A and B connected?"

**Neden DRG Daha İyi:**
- Direct relationship lookup
- Path finding algorithms
- Relationship chains

### 6.3 Sparse Semantic Space

**Örnek:**
- Domain-specific terminology
- Rare entity names
- Technical jargon

**Neden DRG Daha İyi:**
- Graph structure semantic drift'ten etkilenmez
- Entity co-occurrence patterns
- Relationship-based retrieval

### 6.4 When RAG is Better

**Semantic Queries:**
- "Explain the concept of X"
- "What are the implications of Y?"

**Neden RAG Daha İyi:**
- Semantic similarity yeterli
- Graph structure gerekli değil
- Daha hızlı

## 7. Evaluation Methodology

### 7.1 Test Datasets

**3-4 Heterogeneous Datasets:**

1. **Long Narrative Text** (20-page story)
   - Character relationships
   - Plot progression
   - Temporal queries

2. **Factual Text** (Wikipedia biography)
   - Entity relationships
   - Factual queries
   - Cross-references

3. **Technical Document**
   - Code dependencies
   - API relationships
   - Technical concepts

4. **Informal Dialogue** (Chat/Forum)
   - Conversation threads
   - User relationships
   - Topic discussions

### 7.2 Query Types

**Entity-Centric:**
- "What entities are related to X?"
- "Who are the neighbors of Y?"

**Relational:**
- "What is the relationship between X and Y?"
- "How are A and B connected?"

**Semantic:**
- "Explain the concept of X"
- "What are the implications of Y?"

### 7.3 Metrics

**Retrieval Accuracy:**
- Precision@K: Top-K chunk'lar içinde relevant olanlar
- Recall@K: Top-K chunk'lar içinde bulunan relevant chunk'lar
- MRR: İlk relevant chunk'ın rank'ı

**Graph-Specific Metrics:**
- **Path Accuracy**: Bulunan path'lerin doğruluğu
- **Entity Coverage**: Query entity'lerinin graph'da bulunma oranı
- **Relationship Precision**: Bulunan relationship'lerin doğruluğu

**Latency:**
- Query latency (P50, P95)
- Graph traversal time
- Total retrieval time

### 7.4 Comparison Framework

**Baseline:**
- Classic RAG (vector search only)

**Experimental:**
- DRG Search (BFS, DFS, Weighted)
- Hybrid RAG + GraphRAG

**Evaluation:**
- Retrieval accuracy comparison
- Latency comparison
- Cost comparison
- Interpretability comparison

## 8. Implementation Considerations

### 8.1 Graph Indexing

**Node Index:**
- Entity name → Node ID mapping
- Fast lookup için hash index

**Edge Index:**
- Source node → Outgoing edges
- Target node → Incoming edges
- Relation type → Edges

### 8.2 Caching Strategy

**Query Cache:**
- Aynı query'ler için result cache
- Entity extraction cache

**Graph Cache:**
- Frequent traversal paths
- Node embeddings cache

### 8.3 Performance Optimization

**Early Stopping:**
- Semantic similarity threshold
- Maximum hop limit
- Top-K limit

**Parallel Traversal:**
- Multiple seed entities'den parallel traversal
- Multi-threading

**Batch Processing:**
- Multiple queries'leri batch'te işle
- Graph traversal optimization

## 9. Trade-off Özeti

| Algorithm | Precision | Recall | Latency | Complexity |
|-----------|-----------|--------|---------|------------|
| BFS | Medium | High | Medium | Low |
| DFS | High | Medium | Low | Medium |
| Weighted | Very High | High | High | High |
| Multi-Hop | Very High | Very High | Very High | Very High |

**Öneri:** Production için weighted search, research için multi-hop reasoning.

## 10. Future Directions

### 10.1 Learned Graph Traversal

- Reinforcement learning ile optimal traversal path'leri öğren
- Query type'a göre adaptive traversal

### 10.2 Graph Embeddings

- Node embeddings (Node2Vec, GraphSAGE)
- Graph structure'ı embedding space'e map et

### 10.3 Dynamic Graph Updates

- Real-time graph updates
- Incremental graph construction
- Graph versioning

