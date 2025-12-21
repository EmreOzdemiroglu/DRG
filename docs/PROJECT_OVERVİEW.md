# DRG Proje Genel Bakışı

## Proje Adı

**Declarative Relationship Generation (DRG): A DSPy-Inspired Agentic Coding Library for Knowledge Graphs**

## Proje Özeti (Abstract)

Declarative programming, AI sistemlerinin oluşturulma şeklini değiştiriyor. Modellere bir dizi adımı takip etmelerini söylemek yerine, bir geliştirici sistemin ne yapması gerektiğini tanımlar ve bir optimizer bunun nasıl en iyi şekilde çalıştırılabileceğini çıkarır. İşte bu felsefe, önerilen projenin temelini oluşturur: **Declarative Relationship Generation** - yapılandırılmamış verilerden yapılandırılmış Knowledge Graph'lar oluşturmak için DSPy'den ilham alan agentic coding kütüphanesi.

DRG'deki temel yenilik, **declarative doğasıdır**: Entity'leri, ilişkileri ve yapısal mantığı declarative - ancak algoritmik olmayan - bir şekilde tanımlar. Somut olarak, bir geliştirici manuel olarak herhangi bir extraction veya linking algoritması yazmak zorunda değildir. Bu tanımlamalar, bir optimizasyon sürecine signature'lar olarak hareket eder. DRG, bu optimizer'ları declarative bir framework içinde otomatik olarak çalıştırır, geliştiricinin şemasına göre akıl yürütme, iyileştirme ve graph oluşturmayı mümkün kılar.

Diğer otomatik KG sistemlerinin (örneğin stair-lab/kg-gen) aksine, ilişkileri implicit olarak üretirken, DRG **explicit, geliştirici kontrollü bir süreç** uygular. DRG, DSPy'nin benimsediği optimizer felsefesini takip eder - GEPA'ya çok benzer şekilde - entity linking ve ilişki doğruluğunun iteratif iyileştirmesi için, tam şeffaflık ve kontrolü korurken.

Projenin tasarımı, Cursor ve Windsurf gibi agentic coding araçlarının DRG ile entegrasyonunu mümkün kılar ve onlara yapılandırılmış declarative reasoning yetenekleri sağlar; bu, bugünün AI geliştirme manzarasındaki önemli bir eksik boşluğu temsil eder. DRG, açık kaynak AI mühendisliğinde declarative knowledge reasoning'ın temel katmanını oluşturacaktır.

## Temel Felsefe: Declarative Programming

### Ne Yapılacağını Tanımla, Nasıl Yapılacağını Optimizer Çıkarsın

DRG'nin temel felsefesi, **declarative programming** yaklaşımına dayanır:

```python
# Developer sadece NE yapılacağını tanımlar
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# DRG optimizer NASIL yapılacağını çıkarır
entities, triples = extract_typed(text, schema)
```

**Geleneksel Yaklaşım (Imperative):**
- Developer extraction algoritması yazar
- Manual parsing ve linking logic
- Hard-coded rules ve patterns
- Domain-specific kod

**DRG Yaklaşımı (Declarative):**
- Developer sadece schema tanımlar
- Optimizer extraction stratejisini öğrenir
- DSPy signatures otomatik oluşturulur
- Dataset-agnostic, domain-independent

## Temel Özellikler

### 1. Explicit, Developer-Controlled Process

DRG, implicit KG generation sistemlerinin aksine, **explicit kontrol** sağlar:

- **Schema-Based Control**: Developer entity ve relation tiplerini tam olarak tanımlar
- **Transparency**: Her extraction adımı izlenebilir ve açıklanabilir
- **Iterative Refinement**: Optimizer ile sürekli iyileştirme, ancak developer kontrolünde
- **No Black Box**: Tüm süreç şeffaf ve kontrol edilebilir

**Karşılaştırma:**

| Özellik | Implicit Systems (kg-gen) | DRG (Explicit) |
|---------|--------------------------|----------------|
| Control | Black box, implicit rules | Explicit schema, full control |
| Transparency | Limited visibility | Full traceability |
| Refinement | Static, hard to improve | Iterative, optimizer-driven |
| Developer Experience | Limited customization | Full declarative control |

### 2. DSPy Optimizer Philosophy

DRG, DSPy'nin optimizer felsefesini benimser:

- **BootstrapFewShot**: Self-bootstrapping ile hızlı iyileştirme
- **MIPRO**: Multi-prompt optimization için yüksek kalite
- **COPRO**: Compositional optimization için kompleks görevler
- **Iterative Learning**: Training examples ile sürekli iyileştirme

**Optimizer Entegrasyonu:**

```python
from drg import create_optimizer, DRGSchema

# Optimizer oluştur
optimizer = create_optimizer(schema, optimizer_type="bootstrap_few_shot")

# Training examples ekle
optimizer.add_training_example(
    text="Apple produces iPhone.",
    expected_entities=[("Apple", "Company"), ("iPhone", "Product")],
    expected_relations=[("Apple", "produces", "iPhone")]
)

# Optimize et
optimized_extractor = optimizer.optimize()

# Test et ve karşılaştır
comparison = optimizer.compare_before_after(test_examples)
```

### 3. Agentic Coding Tools Integration

DRG, modern agentic coding araçlarıyla entegrasyon için tasarlanmıştır:

- **MCP (Model Context Protocol) API**: Cursor, Windsurf gibi araçlarla entegrasyon
- **Declarative Reasoning**: AI agent'larına structured reasoning yetenekleri
- **Programmatic Interface**: Agent'ların DRG'yi programatik olarak kullanması

**MCP API Örneği:**

```python
from drg.mcp_api import DRGMCPAPI, MCPRequest

# MCP API instance oluştur
api = DRGMCPAPI()

# Schema tanımla
request = MCPRequest(
    method="drg/define_schema",
    params={
        "schema": {
            "entities": ["Company", "Product"],
            "relations": [{"name": "produces", "source": "Company", "target": "Product"}]
        }
    }
)

response = api.handle_request(request)
```

### 4. Foundational Layer for Open-Source AI Engineering

DRG, açık kaynak AI mühendisliği için **temel katman** olarak konumlandırılmıştır:

- **Declarative Knowledge Reasoning**: Structured knowledge için declarative yaklaşım
- **Research-Grade**: Akademik araştırma ve yayın için uygun
- **Community-Ready**: Açık kaynak topluluk için hazır
- **Extensible**: Yeni use case'ler için kolay genişletilebilir

## Mimari Yapı

### Monolithic-Modular Mimarisi

DRG, **monolithic-modular** bir mimari kullanır:

- **Monolithic**: Tüm bileşenler aynı codebase içinde, tek deployment unit
- **Modular**: Her bileşen bağımsız interface'ler üzerinden iletişim kurar
- **Loose Coupling**: Minimal bağımlılıklar, açıkça tanımlı arayüzler
- **High Cohesion**: İlgili fonksiyonellik aynı modülde gruplanır

### Dataset-Agnostic Tasarım

DRG, herhangi bir veri kaynağından bağımsız olarak çalışır:

- **Abstraction Layers**: Veri kaynağı, chunking stratejisi ve embedding modeli arasında net arayüzler
- **Pluggable Components**: Her bileşen bağımsız olarak değiştirilebilir
- **Metadata Preservation**: Her chunk, orijin veri kaynağı ve işlem geçmişi hakkında zengin metadata taşır
- **Domain Adaptation**: Domain-specific optimizasyonlar, core pipeline'ı değiştirmeden eklenebilir

## Bileşenler ve Modüller

### 1. Schema Layer (Declarative Definition)

**Dosyalar:** `drg/schema.py`, `drg/graph/schema_generator.py`

- **DRGSchema**: Basit entity ve relation tanımları
- **EnhancedDRGSchema**: Gelişmiş şema (EntityType, RelationGroup, EntityGroup)
- **DatasetAgnosticSchemaGenerator**: Otomatik şema oluşturma

### 2. Extraction Layer (DSPy-Based)

**Dosyalar:** `drg/extract.py`

- **KGExtractor**: DSPy module, schema'dan dinamik signature'lar oluşturur
- **extract_typed()**: Entity ve relation extraction
- **Tamamen Declarative**: Manuel parsing yok, DSPy otomatik yapar

### 3. Optimizer Layer (Iterative Learning)

**Dosyalar:** `drg/optimizer/optimizer.py`, `drg/optimizer/metrics.py`

- **DRGOptimizer**: DSPy optimizer wrapper
- **Optimizer Types**: BootstrapFewShot, MIPRO, COPRO, LabeledFewShot
- **Evaluation Metrics**: Precision, Recall, F1, Accuracy
- **Iterative Improvement**: Training examples ile sürekli iyileştirme

### 4. Knowledge Graph Layer

**Dosyalar:** `drg/graph.py`, `drg/graph/kg_core.py`

- **KG**: Basit knowledge graph temsili
- **EnhancedKG**: Gelişmiş KG (KGNode, KGEdge, Cluster)
- **Relationship Model**: Enriched relationships, taxonomy
- **Graph Storage**: NetworkX, Neo4j, ArangoDB desteği

### 5. Semantic Pipeline (RAG + GraphRAG)

**Dosyalar:** 
- `drg/chunking/`: Chunking strategies
- `drg/embedding/`: Embedding providers
- `drg/vector_store/`: Vector store abstraction
- `drg/retrieval/`: RAG, DRG Search, Hybrid retrieval

- **Chunking**: Token-based, sentence-based, semantic chunking
- **Embedding**: OpenAI, Gemini, OpenRouter, Local models
- **Vector Store**: Chroma, Qdrant, Pinecone, FAISS
- **Retrieval**: Vector similarity, graph traversal, hybrid

### 6. Clustering & Summarization

**Dosyalar:** `drg/clustering/algorithms.py`, `drg/clustering/summarization.py`

- **Clustering**: Louvain, Leiden, Spectral
- **Summarization**: Cluster-based summarization
- **Community Reports**: GraphRAG-style community reports

### 7. MCP API (Agent Interface)

**Dosyalar:** `drg/mcp_api.py`

- **DRGMCPAPI**: MCP-style API wrapper
- **JSON-RPC 2.0**: Standardized request/response format
- **Agent Integration**: Cursor, Windsurf entegrasyonu için

## Kullanım Senaryoları

### Senaryo 1: Basit Knowledge Graph Extraction

```python
from drg import Entity, Relation, DRGSchema, extract_typed, KG

# Declarative schema tanımla
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# Extract (optimizer otomatik çalışır)
text = "Apple produces iPhone. Google develops Android."
entities, triples = extract_typed(text, schema)

# Knowledge Graph oluştur
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

### Senaryo 2: Iterative Learning ile İyileştirme

```python
from drg import create_optimizer, DRGSchema

# Optimizer oluştur
optimizer = create_optimizer(schema)

# Training examples ekle
optimizer.add_training_example(
    text="Apple produces iPhone.",
    expected_entities=[("Apple", "Company"), ("iPhone", "Product")],
    expected_relations=[("Apple", "produces", "iPhone")]
)

# Optimize et
optimized_extractor = optimizer.optimize()

# Before/after karşılaştır
comparison = optimizer.compare_before_after(test_examples)
print(f"F1 Improvement: {comparison['improvement']['f1']:+.3f}")
```

### Senaryo 3: Agentic Coding Tools Entegrasyonu

```python
from drg.mcp_api import DRGMCPAPI, MCPRequest

# MCP API ile agent entegrasyonu
api = DRGMCPAPI()

# Agent, schema tanımlar
request = MCPRequest(
    method="drg/define_schema",
    params={"schema": {...}}
)

# Agent, extraction yapar
request = MCPRequest(
    method="drg/extract",
    params={"text": "...", "schema_id": "schema_1"}
)

response = api.handle_request(request)
```

### Senaryo 4: Semantic Retrieval (RAG + GraphRAG)

```python
from drg import create_rag_retriever, create_drg_search, HybridRetriever

# RAG retriever
rag = create_rag_retriever(vector_store, embedding_provider)

# DRG search (graph-aware)
drg_search = create_drg_search(knowledge_graph, embedding_provider)

# Hybrid (her ikisini birleştir)
hybrid = HybridRetriever(rag, drg_search)

# Query
results = hybrid.retrieve("What products does Apple produce?", top_k=5)
```

## Proje Durumu ve Roadmap

### ✅ Tamamlananlar

1. **Core Extraction Pipeline**
   - Declarative schema system
   - DSPy-based extraction
   - Knowledge graph construction

2. **Optimizer Integration**
   - DSPy optimizer wrapper
   - Multiple optimizer types
   - Evaluation metrics
   - Iterative improvement loop

3. **Semantic Pipeline**
   - Chunking strategies
   - Embedding abstraction
   - Vector store abstraction
   - RAG retrieval
   - DRG search algorithms

4. **MCP API**
   - MCP-style API wrapper
   - Agent interface
   - JSON-RPC 2.0 format

5. **Clustering & Summarization**
   - Multiple clustering algorithms
   - Cluster summarization
   - Community reports

### 🚧 Devam Edenler

1. **Multi-Dataset Evaluation**
   - 3-4 heterojen dataset üzerinde test
   - Domain sensitivity analysis
   - Performance benchmarking

2. **Documentation**
   - Comprehensive API documentation
   - Usage examples
   - Best practices guide

3. **Testing & Quality**
   - Unit tests
   - Integration tests
   - Performance tests

### 📋 Gelecek Geliştirmeler

1. **Advanced Optimizers**
   - Custom optimizer implementations
   - Multi-task learning
   - Online learning

2. **Enhanced Graph Features**
   - Graph embeddings
   - Dynamic graph updates
   - Graph validation

3. **Production Readiness**
   - Performance optimization
   - Scalability improvements
   - Error handling & recovery

4. **Community Features**
   - Schema marketplace
   - Pre-trained optimizers
   - Community contributions

## Teknoloji Stack

- **Core Framework**: Python 3.10+
- **LLM Framework**: DSPy 2.4.0+
- **Graph Processing**: NetworkX, Neo4j (optional)
- **Vector Stores**: Chroma, Qdrant, Pinecone, FAISS
- **Embedding Models**: OpenAI, Gemini, OpenRouter, Local (sentence-transformers)
- **Clustering**: python-louvain, leidenalg, scikit-learn

## Lisans ve Katkıda Bulunma

- **Lisans**: MIT License
- **Katkıda Bulunma**: Community contributions welcome
- **Issue Tracking**: GitHub Issues
- **Documentation**: Comprehensive docs in `docs/` directory

## Referanslar ve İlgili Projeler

- **DSPy**: Stanford's Declarative Self-improving Pythonic system
- **GraphRAG**: Microsoft's Graph-based Retrieval Augmented Generation
- **stair-lab/kg-gen**: Implicit KG generation (karşılaştırma için)
- **GEPA**: Graph Entity Prediction Architecture (benzer optimizer felsefesi)

## Sonuç

DRG, declarative programming felsefesini knowledge graph generation'a uygulayan, DSPy'den ilham alan bir agentic coding kütüphanesidir. Explicit kontrol, tam şeffaflık ve iteratif iyileştirme ile, açık kaynak AI mühendisliği için declarative knowledge reasoning'ın temel katmanını oluşturmayı hedefler.

**Temel Değer Önerisi:**
- Developer sadece **ne yapılacağını** tanımlar
- Optimizer **nasıl yapılacağını** çıkarır
- **Explicit kontrol** ve **tam şeffaflık**
- **Agentic coding tools** ile entegrasyon
- **Açık kaynak AI mühendisliği** için foundational layer

