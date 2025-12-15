# DRG Project Overview

## Proje Amacı

DRG (Dynamic Retrieval Graph), dataset-agnostic bir semantic pipeline'dır. RAG ve GraphRAG experimentation için tasarlanmış, research-grade, community publication-ready bir sistemdir.

## Şu Anki Durum

### ✅ Tamamlananlar

1. **Knowledge Graph Extraction (DRG)**
   - Declarative schema system (`drg/schema.py`)
   - Entity/relation extraction (DSPy-based, `drg/extract.py`)
   - Knowledge graph class (`drg/graph.py`)
   - CLI interface (`drg/cli.py`)

2. **Dokümantasyon**
   - Pipeline overview (`docs/pipeline_overview.md`)
   - Chunking strategy (`docs/chunking_strategy.md`)
   - Semantic retrieval design (`docs/semantic_retrieval_design.md`)
   - DRG search algorithms (`docs/drg_search.md`)
   - Multi-dataset evaluation (`docs/multi_dataset_evaluation.md`)
   - Clustering & summarization (`docs/clustering_summarization.md`)

3. **Organizasyon**
   - Cursor rules (`.cursorrules`)
   - Modüler monolith klasör yapısı planı

### 🚧 Yapılması Gerekenler (KİŞİ 1 - Bu Sprint)

1. **Chunking & Semantic Pipeline (RAG Core)** ⚠️ KRİTİK
   - Chunk-based reading implementasyonu
   - Token-based chunking (512-1024 tokens)
   - Overlap stratejisi (10-20%)
   - Chunk metadata sistemi
   - **ÖNEMLİ**: Knowledge graph context'i chunk-based reading'e entegre etmek

2. **Embedding Abstraction Layer**
   - OpenAI, Gemini, OpenRouter, Local provider'lar
   - Batch embedding support
   - Cost optimization

3. **Vector Store Abstraction**
   - Chroma, Qdrant, Pinecone, FAISS support
   - Metadata indexing
   - Similarity search

4. **RAG Retrieval Layer**
   - Vector similarity search
   - Knowledge graph context loading (bağlam kaybını önlemek için)
   - Metadata-enhanced retrieval
   - Hybrid retrieval

5. **DRG Search Algorithms**
   - BFS, DFS, Weighted search
   - Semantic score + graph distance optimization
   - Multi-hop reasoning

6. **Multi-Dataset Testing**
   - 3-4 heterojen dataset üzerinde test
   - Chunking quality, retrieval accuracy, entity extraction evaluation

7. **Clustering Infrastructure**
   - Louvain, Leiden, Spectral clustering
   - Cluster summarization
   - Community report generation

## Kritik Tasarım Kararları

### Chunk-Based Reading + Knowledge Graph Context

**Problem**: Chunk-based okurken bağlam kaybı olabilir.

**Çözüm**: 
- Her chunk işlenirken, ilgili knowledge graph node'larını ve relationship'leri kontekste yükle
- Semantic retrieval yaparken hem vector similarity hem de graph structure kullan
- Chunk'ları process ederken, o chunk'tan extract edilen entity'lerin graph'taki komşularını da context'e ekle

**Implementation Strategy**:
1. Chunk'ı process et
2. Chunk'tan entity'leri extract et
3. Bu entity'lerin graph'taki relationship'lerini bul
4. İlgili graph subgraph'ını context'e ekle
5. Semantic retrieval yaparken bu context'i kullan

### Dataset-Agnostic Design

- Tüm bileşenler pluggable
- Domain-specific optimizasyonlar core pipeline'ı değiştirmeden eklenebilir
- Metadata preservation: Her chunk, origin dataset ve processing history hakkında bilgi taşır

## Mimari Prensipler

- **Monolithic-Modular**: Tüm bileşenler aynı codebase'de, ama loose coupling
- **Interface-First**: Her bileşen için önce interface, sonra implementation
- **Dependency Injection**: Hard dependencies yok
- **Configuration Management**: Environment variables + config files

## Teknoloji Stack

- **LLM**: DSPy (mevcut), OpenAI, Gemini, OpenRouter
- **Embedding**: OpenAI, Gemini, OpenRouter, Local (sentence-transformers)
- **Vector Store**: Chroma, Qdrant, Pinecone, FAISS
- **Graph**: NetworkX (mevcut), Neo4j (opsiyonel)
- **Clustering**: python-louvain, leidenalg, scikit-learn

## Sprint Hedefleri

### Bu Sprint (KİŞİ 1)

1. ✅ Chunking modülü implementasyonu
2. ✅ Embedding abstraction layer
3. ✅ Vector store abstraction (Chroma ile başla)
4. ✅ RAG retrieval layer (knowledge graph context ile)
5. ✅ Basit DRG search prototype
6. ✅ 1-2 dataset üzerinde test

### Sonraki Sprint

1. Multi-dataset evaluation (3-4 dataset)
2. DRG search algorithms (BFS, DFS, Weighted)
3. Clustering infrastructure
4. Community report generation

## Notlar

- **DeepSeek**: Kütüphaneyi anlamlı hale getiriyor, context'e ekliyor
- **GraphRAG Reference**: GraphRAG ve KGCEN kütüphanelerini incelemek lazım
- **Context Loading**: Chunk-based reading'de knowledge graph context'i mutlaka yüklenmeli

