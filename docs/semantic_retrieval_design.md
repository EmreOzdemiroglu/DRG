# Semantic Retrieval Layer: Dataset-Agnostic RAG Design

## 1. Genel Bakış

Semantic retrieval layer, chunk embedding'lerini kullanarak query'ler için en ilgili chunk'ları bulur. Bu layer, dataset-agnostic olmalı ve farklı embedding model'leri, vector store'lar ve retrieval stratejileri ile çalışabilmelidir.

## 2. Embedding Abstraction Layer

### 2.1 Provider Comparison

#### 2.1.1 OpenAI Embeddings

**Modeller:**
- `text-embedding-3-small`: 1536 dimensions, hızlı, ucuz
- `text-embedding-3-large`: 3072 dimensions, daha yavaş, daha pahalı, daha iyi quality

**Özellikler:**
- **Semantic Consistency**: Yüksek, cross-domain performance iyi
- **Cost**: Token-based pricing, batch processing ile optimize edilebilir
- **Latency**: Düşük (API-based, network latency)
- **Portability**: Model lock-in riski var, OpenAI API'ye bağımlı

**Kullanım Senaryoları:**
- Production systems (cost-effective)
- Cross-domain applications
- Real-time retrieval

#### 2.1.2 Gemini Embeddings

**Modeller:**
- `embedding-001`: 768 dimensions (genellikle)

**Özellikler:**
- **Semantic Consistency**: İyi, farklı semantic space (OpenAI'den farklı)
- **Cost**: Token-based pricing, OpenAI ile karşılaştırılabilir
- **Latency**: Düşük (API-based)
- **Portability**: Google API'ye bağımlı

**Kullanım Senaryoları:**
- Multi-provider redundancy
- Cost comparison experiments
- Provider diversity

#### 2.1.3 OpenRouter

**Özellikler:**
- **Unified API**: Multiple embedding model'leri tek API'den
- **Model Selection**: Provider'dan bağımsız model seçimi
- **Cost**: Provider-specific pricing
- **Latency**: Provider-dependent

**Kullanım Senaryoları:**
- Model comparison experiments
- Provider-agnostic systems
- Cost optimization

#### 2.1.4 Local Models

**Modeller:**
- `sentence-transformers/all-MiniLM-L6-v2`: 384 dimensions, hızlı
- `sentence-transformers/all-mpnet-base-v2`: 768 dimensions, daha iyi quality

**Özellikler:**
- **Semantic Consistency**: Model-dependent, genellikle OpenAI'den düşük
- **Cost**: Sıfır (local computation)
- **Latency**: Yüksek (local inference), batch processing ile optimize edilebilir
- **Portability**: Tam kontrol, no lock-in

**Kullanım Senaryoları:**
- Privacy-sensitive applications
- Cost-sensitive scenarios
- Offline systems

### 2.2 Embedding Abstraction Interface

**Provider Interface:**

```
EmbeddingProvider:
  - embed(text: str) -> List[float]
  - embed_batch(texts: List[str]) -> List[List[float]]
  - get_dimension() -> int
  - get_model_name() -> str
  - get_cost_per_token() -> float
```

**Provider Selection Strategy:**

1. **Cost Optimization**: Local models → OpenRouter → Provider-specific
2. **Quality Optimization**: OpenAI large → OpenAI small → Local models
3. **Latency Optimization**: Local (batch) → API (streaming)
4. **Portability**: Local → OpenRouter → Provider-specific

### 2.3 Semantic Consistency Analysis

**Cross-Domain Performance:**

- **Domain Adaptation**: Embedding model'leri farklı domain'lerde farklı performans gösterir
- **Semantic Drift**: Model update'lerinde semantic space değişebilir
- **Dimension Mismatch**: Farklı model'ler farklı dimension'larda embed eder

**Mitigation Strategies:**

- **Model Versioning**: Embedding model version'ını metadata'da sakla
- **Re-embedding Strategy**: Model değiştiğinde re-embedding pipeline'ı
- **Dimension Normalization**: Farklı dimension'ları normalize et (PCA, etc.)

### 2.4 Cost Analysis

**Cost Factors:**

- **Token Count**: Embedding cost token count'a bağlı
- **Batch Size**: Batch processing cost'u düşürür
- **Caching**: Aynı text'i tekrar embed etme

**Optimization Strategies:**

- **Batch Processing**: Mümkün olduğunca batch embed et
- **Caching Layer**: Embedding cache (text → embedding mapping)
- **Incremental Updates**: Sadece yeni chunk'ları embed et

## 3. Vector Store Abstraction

### 3.1 Vector Store Options

#### 3.1.1 Chroma

**Özellikler:**
- **Embedding Storage**: Native vector storage
- **Metadata Indexing**: Rich metadata support
- **Hybrid Search**: Vector + metadata filtering
- **Scalability**: Local-first, distributed support

**Kullanım Senaryoları:**
- Development/prototyping
- Small to medium datasets
- Metadata-rich applications

#### 3.1.2 Qdrant

**Özellikler:**
- **Performance**: Yüksek performans, production-ready
- **Scalability**: Distributed, horizontal scaling
- **Advanced Features**: Payload filtering, geo-search
- **Cost**: Self-hosted (free) veya cloud (paid)

**Kullanım Senaryoları:**
- Production systems
- Large-scale datasets
- High-performance requirements

#### 3.1.3 Pinecone

**Özellikler:**
- **Managed Service**: Fully managed, no infrastructure
- **Scalability**: Automatic scaling
- **Cost**: Pay-per-use, can be expensive at scale
- **Features**: Metadata filtering, hybrid search

**Kullanım Senaryoları:**
- Quick prototyping
- Managed infrastructure preference
- Variable workload

#### 3.1.4 FAISS (Local)

**Özellikler:**
- **Performance**: Yüksek performans, Facebook research
- **Scalability**: Memory-based, limited by RAM
- **Cost**: Free, self-hosted
- **Features**: Basic vector search, limited metadata

**Kullanım Senaryoları:**
- Research/experimentation
- Small to medium datasets
- Cost-sensitive applications

### 3.2 Vector Store Interface

**Store Interface:**

```
VectorStore:
  - add(embeddings: List[List[float]], metadata: List[Dict]) -> List[str]
  - search(query_embedding: List[float], k: int, filters: Dict) -> List[Dict]
  - update(id: str, embedding: List[float], metadata: Dict)
  - delete(ids: List[str])
  - get_metadata(id: str) -> Dict
```

**Metadata Schema:**

```json
{
  "chunk_id": "dataset_doc_001_chunk_000",
  "sequence_index": 0,
  "origin_dataset": "dataset_name",
  "semantic_tags": {...},
  "embedding_model": "openai/text-embedding-3-small"
}
```

## 4. Retrieval Strategies

### 4.1 Vector Similarity Search (Varsayılan)

**Algoritma:**

1. Query'yi embed et
2. Vector store'da similarity search yap (cosine similarity)
3. Top-K chunk'ları döndür
4. Metadata filtering uygula (opsiyonel)

**Scoring:**

```
score = cosine_similarity(query_embedding, chunk_embedding)
```

**Avantajlar:**
- Hızlı
- Basit
- Semantic similarity'ye dayalı

**Dezavantajlar:**
- Graph structure'ı kullanmaz
- Multi-hop reasoning yok
- Entity relationships ignore edilir

### 4.2 Metadata-Enhanced Retrieval

**Algoritma:**

1. Query'yi embed et
2. Query'den entity'leri extract et
3. Metadata filtering uygula (entity, topic, etc.)
4. Filtered set'te similarity search yap
5. Top-K chunk'ları döndür

**Scoring:**

```
base_score = cosine_similarity(query_embedding, chunk_embedding)
metadata_boost = calculate_metadata_match(query_metadata, chunk_metadata)
final_score = base_score * (1 + metadata_boost)
```

**Avantajlar:**
- Entity-aware retrieval
- Topic filtering
- Precision artışı

**Dezavantajlar:**
- Entity extraction cost'u
- Metadata quality'ye bağımlı

### 4.3 Hybrid Retrieval (Vector + Keyword)

**Algoritma:**

1. Query'yi embed et
2. Vector similarity search yap
3. Keyword-based search yap (BM25, etc.)
4. İki sonucu birleştir (reciprocal rank fusion)
5. Top-K chunk'ları döndür

**Scoring:**

```
vector_score = cosine_similarity(query_embedding, chunk_embedding)
keyword_score = bm25_score(query_keywords, chunk_text)
final_score = α * vector_score + (1 - α) * keyword_score
```

**Avantajlar:**
- Exact match support
- Semantic + lexical combination
- Better recall

**Dezavantajlar:**
- Keyword index gerektirir
- Parameter tuning (α)
- Daha yavaş

### 4.4 Re-ranking Strategy

**Algoritma:**

1. Initial retrieval (vector search, top-100)
2. Re-rank using cross-encoder (BERT, etc.)
3. Top-K chunk'ları döndür

**Scoring:**

```
initial_score = cosine_similarity(query_embedding, chunk_embedding)
rerank_score = cross_encoder_score(query_text, chunk_text)
final_score = β * initial_score + (1 - β) * rerank_score
```

**Avantajlar:**
- Precision artışı
- Query-chunk interaction modeling

**Dezavantajlar:**
- Yavaş (cross-encoder inference)
- Cost yüksek
- Batch processing gerekli

## 5. Multi-Stage Retrieval

### 5.1 Coarse-to-Fine Retrieval

**Stage 1: Coarse Retrieval**
- Vector similarity search, top-100
- Fast, low precision

**Stage 2: Fine Retrieval**
- Re-ranking with cross-encoder, top-10
- Slow, high precision

**Avantajlar:**
- Balanced latency/precision
- Cost optimization (sadece top-100'ü re-rank et)

### 5.2 Query Expansion

**Algoritma:**

1. Original query'yi embed et
2. Query expansion (synonyms, related terms)
3. Expanded query'leri embed et
4. Multi-query retrieval (OR fusion)
5. Top-K chunk'ları döndür

**Avantajlar:**
- Recall artışı
- Query understanding improvement

**Dezavantajlar:**
- Query expansion cost'u
- Noise riski

## 6. Evaluation Metrics

### 6.1 Retrieval Accuracy

**Precision@K:**
- Top-K chunk'lar içinde relevant olanların oranı
- Target: Precision@10 > 0.7

**Recall@K:**
- Top-K chunk'lar içinde bulunan relevant chunk'ların tüm relevant chunk'lara oranı
- Target: Recall@10 > 0.6

**MRR (Mean Reciprocal Rank):**
- İlk relevant chunk'ın rank'ının reciprocal'i
- Target: MRR > 0.8

### 6.2 Latency Metrics

**P50 Latency:**
- Median query latency
- Target: < 100ms (vector search only)

**P95 Latency:**
- 95th percentile latency
- Target: < 500ms (with re-ranking)

### 6.3 Cost Metrics

**Cost per Query:**
- Embedding cost + vector search cost
- Target: < $0.001 per query (OpenAI small)

## 7. Domain-Specific Adaptations

### 7.1 Long Narrative Text

**Challenges:**
- Temporal queries ("what happened after X?")
- Character queries ("what did character Y do?")

**Adaptations:**
- Sequence index filtering
- Character entity filtering
- Temporal metadata (if available)

### 7.2 Factual Text

**Challenges:**
- Factual accuracy
- Entity consistency

**Adaptations:**
- Entity-centric retrieval
- Fact verification support
- Cross-reference resolution

### 7.3 Technical Documents

**Challenges:**
- Code block retrieval
- Table retrieval
- Cross-reference resolution

**Adaptations:**
- Code-aware chunking (atomic code blocks)
- Table-aware retrieval
- Reference resolution

### 7.4 Informal Dialogue

**Challenges:**
- Context switching
- Multi-party dialogue
- Temporal ordering

**Adaptations:**
- Speaker attribution filtering
- Conversation turn ordering
- Context window expansion

## 8. Trade-off Özeti

| Strateji | Precision | Recall | Latency | Cost |
|----------|-----------|--------|---------|------|
| Vector Search | Medium | High | Low | Low |
| Metadata-Enhanced | High | Medium | Low | Medium |
| Hybrid | High | High | Medium | Medium |
| Re-ranking | Very High | High | High | High |

**Öneri:** Production için metadata-enhanced retrieval, research için hybrid + re-ranking kombinasyonu.

## 9. Implementation Considerations

### 9.1 Caching Strategy

- **Query Embedding Cache**: Aynı query'yi tekrar embed etme
- **Retrieval Result Cache**: Frequent query'ler için result cache
- **Metadata Cache**: Metadata lookup'ları cache'le

### 9.2 Batch Processing

- **Batch Embedding**: Mümkün olduğunca batch embed et
- **Batch Retrieval**: Multiple query'leri batch'te işle
- **Async Processing**: Non-blocking retrieval

### 9.3 Error Handling

- **Embedding Failures**: Retry logic, fallback provider
- **Vector Store Failures**: Retry logic, fallback store
- **Timeout Handling**: Query timeout, partial results

