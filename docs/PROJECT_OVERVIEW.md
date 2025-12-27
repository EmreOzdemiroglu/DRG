# DRG Projesi: Kapsamlı Proje Dokümantasyonu

## 📋 İçindekiler

1. [Proje Genel Bakış](#proje-genel-bakış)
2. [Temel Kavramlar](#temel-kavramlar)
3. [Proje Felsefesi ve Tasarım Prensipleri](#proje-felsefesi-ve-tasarım-prensipleri)
4. [Proje Yapısı](#proje-yapısı)
5. [Declarative Yapı](#declarative-yapı)
6. [Pipeline Mimarisi](#pipeline-mimarisi)
7. [Bileşenler ve Metodlar](#bileşenler-ve-metodlar)
8. [Diğer KG Sistemlerinden Farkları](#diğer-kg-sistemlerinden-farkları)
9. [Kullanım Örnekleri](#kullanım-örnekleri)
10. [Geliştirme ve Katkıda Bulunma](#geliştirme-ve-katkıda-bulunma)

---

## Proje Genel Bakış

### DRG Nedir?

**DRG (Declarative Relationship Generation)**, metinlerden bilgi grafiği (Knowledge Graph) çıkarımı yapmak için tasarlanmış, dataset-agnostic (veri kaynağından bağımsız) bir Python kütüphanesidir. DRG, modern Large Language Model (LLM) teknolojilerini kullanarak, sadece şema tanımlayarak otomatik olarak entity (varlık) ve relation (ilişki) çıkarımı yapar.

### Projenin Temel Amacı

DRG projesi, aşağıdaki temel amaçlara hizmet eder:

1. **Declarative (Deklaratif) Yaklaşım**: Kullanıcılar "ne" istediklerini tanımlar, sistem "nasıl" yapılacağını otomatik olarak halleder.

2. **Dataset-Agnostic Tasarım**: Herhangi bir veri kaynağından (metin, PDF, JSON, vb.) bağımsız olarak çalışır.

3. **GraphRAG Desteği**: Sadece klasik RAG (Retrieval-Augmented Generation) değil, aynı zamanda GraphRAG (Graph-based RAG) yapısını da destekler.

4. **Research-Grade Kalite**: Akademik araştırmalar ve yayınlar için uygun, yüksek kaliteli kod yapısı.

5. **Community Publication-Ready**: Topluluk tarafından kullanılmaya ve yayınlanmaya hazır bir sistem.

### Projenin Özellikleri

- ✅ **Declarative Schema**: Sadece entity tipleri ve ilişkileri tanımlayın, gerisini DRG halletsin
- ✅ **Otomatik Schema Generation**: Metinden otomatik olarak EnhancedDRGSchema oluşturma (`generate_schema_from_text`)
- ✅ **DSPy Entegrasyonu**: Modern LLM'lerle çalışan güçlü extraction pipeline
- ✅ **Enhanced Schema**: EntityType (properties, examples), RelationGroup (semantic grouping), Relation (description, detail) ile zengin şema tanımları
- ✅ **Chunk-based KG Extraction**: Her chunk üzerinde bağımsız extraction, sonuçların birleştirilmesi
- ✅ **Schema Validation**: Extraction sonuçlarının şemaya uygunluğunun otomatik kontrolü
- ✅ **Otomatik LLM Konfigürasyonu**: Environment variable'lardan otomatik model ve API key yönetimi
- ✅ **GraphRAG Pipeline**: Chunking → KG Extraction → Embedding → Clustering → Community Reports → Retrieval
- ✅ **Clustering Desteği**: Louvain, Leiden, Spectral algoritmaları ile community detection
- ✅ **Community Reports**: Her cluster için otomatik özet raporlar (top actors, top relationships, themes)
- ✅ **Preset-based Chunking**: "graphrag" gibi preset'lerle kolay chunking konfigürasyonu
- ✅ **Multi-Provider Desteği**: OpenAI, Gemini, Anthropic, OpenRouter, Perplexity, Ollama
- ✅ **FastAPI Web Server**: RESTful API ve interaktif web UI ile KG görselleştirme
- ✅ **Graph Visualization**: Cytoscape.js tabanlı interaktif graph görselleştirme (zoom, pan, community coloring)
- ✅ **Query Provenance**: Query → chunks → community → summary → answer provenance chain tracking
- ✅ **Neo4j Integration**: Knowledge graph'ı Neo4j'e senkronize etme ve persistence
- ✅ **MCP API**: Agent interface için Model Context Protocol desteği
- ✅ **Optimizer Desteği**: DSPy optimizer'ları ile iterative learning
- ✅ **Self-loop Filtering**: KG'de self-loop edge'lerin otomatik filtrelenmesi
- ✅ **Isolated Node Filtering**: Visualization'da bağlantısız node'ların otomatik filtrelenmesi

---

## Temel Kavramlar

### Knowledge Graph (Bilgi Grafiği) Nedir?

**Knowledge Graph (KG)**, bilgileri yapılandırılmış bir şekilde temsil eden bir graf yapısıdır. KG'de:

- **Node (Düğüm)**: Entity'ler (varlıklar) - örneğin: "Apple", "Steve Jobs", "iPhone"
- **Edge (Kenar)**: Relation'lar (ilişkiler) - örneğin: "Apple → produces → iPhone"

KG'ler, bilgileri ilişkisel bir yapıda sakladığı için, sadece metin aramasından daha güçlü sorgulama ve çıkarım yapılmasına olanak tanır.

### Entity (Varlık) Nedir?

**Entity**, metinde bahsedilen somut veya soyut kavramlardır. Örneğin:
- **Kişiler**: "Steve Jobs", "Tim Cook"
- **Şirketler**: "Apple Inc.", "Google"
- **Ürünler**: "iPhone", "iPad"
- **Lokasyonlar**: "Cupertino", "California"

### Relation (İlişki) Nedir?

**Relation**, iki entity arasındaki bağlantıyı temsil eder. Örneğin:
- "Apple → produces → iPhone" (Apple, iPhone üretir)
- "Steve Jobs → founded_by → Apple" (Steve Jobs, Apple'ı kurdu)
- "Tim Cook → ceo_of → Apple" (Tim Cook, Apple'ın CEO'sudur)

### Chunking (Parçalama) Nedir?

**Chunking**, uzun metinleri daha küçük, işlenebilir parçalara bölme işlemidir. LLM'ler genellikle sınırlı token kapasitesine sahip olduğu için, uzun metinler önce chunk'lara bölünür, sonra her chunk üzerinde işlem yapılır.

DRG'de chunking stratejileri:
- **Token-based**: Token sayısına göre bölme
- **Sentence-based**: Cümle sınırlarına göre bölme
- **Semantic**: Anlamsal benzerliğe göre bölme

### Embedding (Vektörleştirme) Nedir?

**Embedding**, metinleri sayısal vektörlere dönüştürme işlemidir. Bu vektörler, metinlerin anlamsal benzerliğini ölçmek için kullanılır. Örneğin, "Apple" ve "iPhone" kelimeleri birbirine yakın vektörlerle temsil edilir.

### RAG (Retrieval-Augmented Generation) Nedir?

**RAG**, LLM'lerin bilgiyi gerçek zamanlı olarak retrieve (getirme) edip kullanmasına olanak tanıyan bir yaklaşımdır. RAG'de:
1. Metinler chunk'lara bölünür ve embed edilir
2. Query (sorgu) embed edilir
3. Query'ye en benzer chunk'lar bulunur (vector similarity)
4. Bu chunk'lar LLM'e context olarak verilir
5. LLM, bu context'i kullanarak cevap üretir

### GraphRAG Nedir?

**GraphRAG**, RAG'in gelişmiş bir versiyonudur. GraphRAG'de:
1. Metinlerden Knowledge Graph oluşturulur
2. Query'den seed entity'ler bulunur (embedding kullanarak)
3. Graph traversal (graf gezinme) ile ilgili entity'ler bulunur
4. Community reports (topluluk raporları) oluşturulur
5. Bu bilgiler LLM'e context olarak verilir

GraphRAG'ın avantajları:
- Multi-hop reasoning (çok adımlı çıkarım) yapabilir
- Entity relationships explicit olarak kullanılır
- Graph topology'den bilgi çıkarılır

### Declarative (Deklaratif) Programlama Nedir?

**Declarative programming**, "ne" istediğinizi tanımladığınız, sistemin "nasıl" yapılacağını otomatik olarak hallettiği bir programlama paradigmasıdır.

**Imperative (Emirsel) Yaklaşım** (Geleneksel):
```python
# Nasıl yapılacağını adım adım tanımlarsınız
text = "Apple produces iPhone"
# 1. Metni parse et
# 2. Entity'leri bul
# 3. Relation'ları bul
# 4. KG oluştur
```

**Declarative (Deklaratif) Yaklaşım** (DRG):
```python
# Sadece ne istediğinizi tanımlarsınız
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
# Sistem otomatik olarak extraction yapar
entities, triples = extract_typed(text, schema)
```

---

## Proje Felsefesi ve Tasarım Prensipleri

### 1. Monolithic-Modular Mimarisi

DRG, **monolithic-modular** bir mimari kullanır:

- **Monolithic**: Tüm bileşenler aynı codebase içinde, tek bir deployment unit
- **Modular**: Her bileşen bağımsız interface'ler üzerinden iletişim kurar
- **Loose Coupling**: Bileşenler arası bağımlılıklar minimal ve açıkça tanımlıdır
- **High Cohesion**: İlgili fonksiyonellik aynı modülde gruplanır

Bu yaklaşımın avantajları:
- Kolay deployment (tek bir paket)
- Modüler test edilebilirlik
- Esnek bileşen değişimi

### 2. Dataset-Agnostic Tasarım

DRG, herhangi bir veri kaynağından bağımsız olarak çalışır:

- **Abstraction Layers**: Veri kaynağı, chunking stratejisi ve embedding modeli arasında net arayüzler
- **Pluggable Components**: Her bileşen bağımsız olarak değiştirilebilir ve test edilebilir
- **Metadata Preservation**: Her chunk, orijin veri kaynağı ve işlem geçmişi hakkında zengin metadata taşır
- **Domain Adaptation**: Domain-specific optimizasyonlar, core pipeline'ı değiştirmeden eklenebilir

### 3. Interface-First Design

Her bileşen için önce interface tanımlanır, sonra implementation yapılır:

```python
# Interface
class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

# Implementation
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> List[float]:
        # Implementation
        pass
```

### 4. Dependency Injection

Hard dependencies yerine dependency injection kullanılır:

```python
# Bad
class Chunker:
    def __init__(self):
        self.tokenizer = TiktokenTokenizer()  # Hard dependency

# Good
class Chunker:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer  # Injected dependency
```

---

## Proje Yapısı

### Klasör Yapısı

Proje yapısı, **monolithic-modular** mimari prensibine uygun olarak düzenlenmiştir. Tüm kod `drg/` modülü altında toplanmış, ancak her bileşen bağımsız modüller halinde organize edilmiştir.

```
DRG/                                    # Proje root dizini
│
├── 📦 drg/                             # Ana Python modülü (Core Library)
│   │
│   ├── 🎯 Core Components              # Temel bileşenler
│   │   ├── __init__.py                 # Public API exports
│   │   ├── schema.py                   # Schema tanımları (Entity, Relation, DRGSchema, EnhancedDRGSchema)
│   │   ├── extract.py                  # DSPy-based KG extraction (KGExtractor, generate_schema_from_text)
│   │   ├── graph.py                    # Legacy KG class (geriye dönük uyumluluk için)
│   │   └── cli.py                      # Komut satırı arayüzü
│   │
│   ├── ✂️  chunking/                   # Metin Parçalama Katmanı
│   │   ├── __init__.py
│   │   ├── strategies.py               # Chunking stratejileri (token, sentence, semantic)
│   │   └── validators.py               # Chunk doğrulama ve validasyon
│   │
│   ├── 🧮 embedding/                   # Embedding Sağlayıcıları
│   │   ├── __init__.py
│   │   ├── providers.py                # Embedding provider interface
│   │   ├── openai.py                   # OpenAI embedding provider
│   │   ├── gemini.py                   # Google Gemini embedding provider
│   │   ├── openrouter.py               # OpenRouter embedding provider
│   │   └── local.py                    # Local model provider (sentence-transformers)
│   │
│   ├── 💾 vector_store/                # Vektör Veritabanı Soyutlaması
│   │   ├── __init__.py
│   │   ├── interface.py                # Vector store interface
│   │   ├── chroma.py                   # ChromaDB implementation
│   │   ├── qdrant.py                   # Qdrant implementation
│   │   ├── faiss.py                    # FAISS implementation
│   │   └── factory.py                  # Factory pattern for vector stores
│   │
│   ├── 🕸️  graph/                      # Knowledge Graph Katmanı
│   │   ├── __init__.py
│   │   ├── kg_core.py                  # EnhancedKG, KGNode, KGEdge, Cluster sınıfları
│   │   ├── visualization.py            # KG görselleştirme (Mermaid, PyVis)
│   │   ├── visualization_adapter.py    # Web viz adapters (Cytoscape.js, vis-network, D3.js)
│   │   ├── community_report.py         # Community report generation
│   │   ├── neo4j_exporter.py           # Neo4j persistence layer
│   │   ├── schema_generator.py         # Dataset-agnostic schema generation
│   │   └── relationship_model.py       # Relationship type classification
│   │
│   ├── 🔍 retrieval/                   # Retrieval Katmanı
│   │   ├── __init__.py
│   │   ├── rag.py                      # Classic RAG retrieval (vector similarity)
│   │   ├── graphrag.py                 # GraphRAG retrieval (KG traversal + community reports)
│   │   ├── drg_search.py               # DRG search algorithms
│   │   └── hybrid.py                   # Hybrid RAG + GraphRAG retriever
│   │
│   ├── 🔗 clustering/                  # Clustering Katmanı
│   │   ├── __init__.py
│   │   ├── algorithms.py               # Clustering algoritmaları (Louvain, Leiden, Spectral)
│   │   └── summarization.py            # Cluster özetleme (community reports)
│   │
│   ├── 🎛️  optimizer/                  # DSPy Optimizer Modülü
│   │   ├── __init__.py
│   │   ├── optimizer.py                # DRGOptimizer class
│   │   └── metrics.py                  # Evaluation metrics (precision, recall, F1)
│   │
│   ├── 🌐 api/                         # FastAPI Web Server
│   │   ├── __init__.py
│   │   ├── server.py                   # FastAPI app ve REST API endpoints
│   │   ├── templates/                  # HTML templates
│   │   │   └── index.html              # Cytoscape.js interaktif graph visualization UI
│   │   └── static/                     # Static dosyalar (CSS, JavaScript)
│   │
│   └── mcp_api.py                      # Model Context Protocol (MCP) API interface
│
├── 📚 docs/                            # Dokümantasyon (KOD YOK, SADECE MARKDOWN)
│   ├── PROJECT_OVERVIEW.md             # Bu dosya - Kapsamlı proje dokümantasyonu
│   ├── pipeline_overview.md            # Pipeline mimarisi ve akış diyagramları
│   ├── schema_design.md                # Schema tasarım prensipleri
│   ├── chunking_strategy.md            # Chunking stratejileri ve best practices
│   ├── drg_search.md                   # DRG search algoritmaları
│   ├── clustering_summarization.md     # Clustering ve community report generation
│   ├── optimizer_design.md             # DSPy optimizer entegrasyonu
│   ├── relationship_model.md           # Relationship classification modeli
│   └── mcp_integration.md              # MCP API entegrasyonu
│
├── 💡 examples/                        # Kullanım Örnekleri
│   ├── graphrag_pipeline_example.py    # Tam GraphRAG pipeline örneği (Ana örnek)
│   ├── api_server_example.py           # FastAPI server başlatma örneği
│   ├── mcp_demo.py                     # MCP API demo
│   └── optimizer_demo.py               # DSPy optimizer demo
│
├── 🧪 tests/                           # Test Suite
│   ├── test_basic.py                   # Temel testler (tüm provider'lar için)
│   └── multi_dataset/                  # Multi-dataset evaluation
│       └── evaluation.py               # Çoklu veri seti değerlendirme testleri
│
├── 📥 inputs/                          # Giriş Dosyaları (Test Verileri)
│   ├── 1example_text.txt               # Örnek 1: Metin dosyası
│   ├── 1example_schema.json            # Örnek 1: Schema (opsiyonel - yoksa otomatik oluşturulur)
│   ├── 2example_text.txt               # Örnek 2: Metin dosyası
│   ├── 3example_text.txt               # Örnek 3: Metin dosyası
│   ├── 3example_schema.json            # Örnek 3: Schema
│   ├── 4example_text.txt               # Örnek 4: Metin dosyası
│   └── 4example_schema.json            # Örnek 4: Schema
│
├── 📤 outputs/                         # Çıktı Dosyaları (Pipeline Sonuçları)
│   ├── {example_name}_schema.json      # Oluşturulan/güncellenen schema
│   ├── {example_name}_kg.json          # Knowledge Graph (EnhancedKG formatında)
│   ├── {example_name}_community_reports.json  # Community/cluster raporları
│   └── {example_name}_summary.json     # Pipeline özeti (istatistikler)
│
├── 🚀 Scripts                          # Yardımcı Scriptler
│   ├── start_api_server.sh             # API server başlatma scripti (GEMINI_API_KEY otomatik export)
│   └── restart_api_server.sh           # API server yeniden başlatma scripti (port 8000 temizleme)
│
├── 📄 Configuration & Docs             # Konfigürasyon ve Dokümantasyon Dosyaları
│   ├── README.md                       # Proje ana README dosyası
│   ├── README_API.md                   # API server dokümantasyonu
│   ├── SETUP.md                        # Kurulum talimatları
│   ├── QUICK_START.md                  # Hızlı başlangıç rehberi
│   ├── pyproject.toml                  # Proje konfigürasyonu (Python packaging)
│   ├── requirements.txt                # Python bağımlılıkları
│   └── LICENSE                         # Lisans dosyası
│
└── uv.lock                             # UV package manager lock file (opsiyonel)
```

#### Yapı Açıklamaları

**📦 drg/**: Ana Python modülü. Tüm core functionality burada toplanmıştır. Modüler yapı, ancak tek bir paket olarak deploy edilir (monolithic-modular mimari).

**🎯 Core Components**: Schema tanımları, extraction logic, CLI interface gibi temel bileşenler.

**Katmanlar (Layers)**: Pipeline'ın adımlarına karşılık gelen modüller:
- **chunking/**: Metin parçalama
- **embedding/**: Vektörleştirme
- **graph/**: KG oluşturma ve yönetimi
- **retrieval/**: Bilgi erişimi (RAG, GraphRAG)
- **clustering/**: Topluluk tespiti

**🌐 api/**: FastAPI web server ve interaktif graph visualization UI.

**📚 docs/**: Teknik dokümantasyon. Kod içermez, sadece tasarım ve mimari dokümantasyonu.

**💡 examples/**: Kullanım örnekleri. Yeni kullanıcılar için başlangıç noktası.

### Modül Açıklamaları

#### `drg/schema.py`
Schema tanımları için temel sınıflar:
- `Entity`: Basit entity tanımı
- `Relation`: Relation tanımı (name, source, target, description, detail)
  - `description`: Bağlantı sebebi/türü açıklaması
  - `detail`: Bağlantı detayı (tek cümleyle neden bağlantılı olduğu)
- `DRGSchema`: Legacy schema class (backward compatibility)
- `EntityType`: Gelişmiş entity tanımı (name, description, examples, properties)
- `RelationGroup`: İlişkili relation'ları semantic olarak gruplama
- `EnhancedDRGSchema`: Gelişmiş schema class (entity_types, relation_groups, auto_discovery)

#### `drg/extract.py`
DSPy kullanarak entity ve relation extraction:
- `KGExtractor`: Ana extraction class (chunk-based processing için kullanılır)
- `extract_typed()`: Typed entity ve relation extraction
- `extract_triples()`: Sadece relation extraction (backward compatibility)
- `generate_schema_from_text()`: Metinden otomatik EnhancedDRGSchema oluşturma
- `_configure_llm_auto()`: Otomatik LLM konfigürasyonu (OpenRouter, OpenAI, vb.)

#### `drg/chunking/`
Metin parçalama stratejileri:
- `TokenBasedChunker`: Token sayısına göre chunking
- `SentenceBasedChunker`: Cümle sınırlarına göre chunking
- `ChunkValidator`: Chunk doğrulama
- `create_chunker()`: Factory function (preset desteği ile)
  - Preset'ler: "graphrag", "medium"

#### `drg/embedding/`
Embedding provider'ları:
- `OpenAIEmbeddingProvider`: OpenAI embedding'leri
- `GeminiEmbeddingProvider`: Google Gemini embedding'leri
- `OpenRouterEmbeddingProvider`: OpenRouter embedding'leri
- `LocalEmbeddingProvider`: Local model embedding'leri (sentence-transformers)

#### `drg/graph/kg_core.py`
Enhanced Knowledge Graph yapısı:
- `KGNode`: Graph node (id, type, properties, metadata, embedding)
- `KGEdge`: Graph edge (source, target, relationship_type, relationship_detail, metadata)
- `Cluster`: Cluster tanımı (id, node_ids, metadata)
- `EnhancedKG`: Ana KG class (nodes, edges, clusters, community reports)

#### `drg/retrieval/`
Retrieval stratejileri:
- `RAGRetriever`: Classic RAG (vector similarity search)
- `GraphRAGRetriever`: GraphRAG (KG traversal + community reports)
- `DRGSearch`: DRG search algorithms
- `HybridRetriever`: RAG + GraphRAG hybrid

#### `drg/clustering/`
Clustering algoritmaları:
- `LouvainClustering`: Louvain community detection (python-louvain gerekli)
- `LeidenClustering`: Leiden algorithm (leidenalg, python-igraph gerekli)
- `SpectralClustering`: Spectral clustering (scikit-learn gerekli)
- `create_clustering_algorithm()`: Factory function
- EnhancedKG ve NetworkX graph formatlarını destekler
- Self-loop edge'leri otomatik filtreler

#### `drg/optimizer/`
DSPy optimizer desteği:
- `DRGOptimizer`: Optimizer wrapper class
- `ExtractionMetrics`: Evaluation metrics (precision, recall, F1)
- `BootstrapFewShot`, `MIPRO`, `COPRO`, `LabeledFewShot` desteği

---

## Declarative Yapı

### Declarative Yaklaşımın Avantajları

1. **Basitlik**: Kullanıcı sadece "ne" istediğini tanımlar
2. **Esneklik**: Sistem otomatik olarak en iyi yöntemi seçer
3. **Bakım Kolaylığı**: Implementation detayları gizlenir
4. **Test Edilebilirlik**: Schema'lar kolayca test edilebilir

### DRG'de Declarative Yapı

#### 1. Schema Tanımlama

```python
# Basit schema
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# Gelişmiş schema (description'lar ile)
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[
        Relation(
            "produces", 
            "Company", 
            "Product",
            description="Bu ilişki, bir şirketin belirli bir ürünü ürettiğini, geliştirdiğini veya piyasaya sürdüğünü gösterir. Ürün, şirketin ana faaliyet alanı veya üretim hattının bir parçası olabilir."
        )
    ]
)
```

#### 2. JSON Schema Formatı

Schema'lar JSON formatında da tanımlanabilir:

```json
{
  "entities": [
    {
      "name": "Company",
      "description": "Business organizations and corporations"
    },
    {
      "name": "Product",
      "description": "Products, devices, goods"
    }
  ],
  "relations": [
    {
      "name": "produces",
      "source": "Company",
      "target": "Product",
      "description": "Bu ilişki, bir şirketin belirli bir ürünü ürettiğini gösterir..."
    }
  ]
}
```

#### 3. Otomatik Extraction

Schema tanımlandıktan sonra, extraction otomatik olarak yapılır:

```python
# Sadece schema ve metin yeterli
text = "Apple produces iPhone, iPad, and Mac computers."
entities, triples = extract_typed(text, schema)

# Sistem otomatik olarak:
# 1. LLM'i konfigüre eder
# 2. Entity extraction yapar
# 3. Relation extraction yapar
# 4. Sonuçları döndürür
```

### Declarative vs Imperative Karşılaştırma

**Imperative Yaklaşım** (Geleneksel):
```python
# Adım adım manuel işlem
text = "Apple produces iPhone"
# 1. Metni tokenize et
tokens = tokenize(text)
# 2. NER (Named Entity Recognition) çalıştır
entities = ner_model.predict(tokens)
# 3. Relation extraction çalıştır
relations = relation_model.predict(tokens, entities)
# 4. KG oluştur
kg = build_kg(entities, relations)
```

**Declarative Yaklaşım** (DRG):
```python
# Sadece ne istediğinizi tanımlayın
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
# Sistem otomatik olarak halleder
entities, triples = extract_typed(text, schema)
kg = EnhancedKG.from_typed(entities, triples)
```

---

## Pipeline Mimarisi

### Tam GraphRAG Pipeline

DRG'nin tam pipeline'ı şu adımlardan oluşur:

```
┌─────────────────────────────────────────────────────────────┐
│                    1. CHUNKING                              │
│  Metin → Token/Sentence-based Chunking → Chunks            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              2. KNOWLEDGE GRAPH EXTRACTION                  │
│  Chunks → DSPy Extraction → Entities + Relations            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             3. ENHANCED KG OLUŞTURMA                        │
│  Entities + Relations → KGNode + KGEdge → EnhancedKG       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           4. ENTITY EMBEDDING'LERİ EKLEME                    │
│  EnhancedKG → Embedding Provider → Node Embeddings         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        5. CLUSTERING VE COMMUNITY REPORTS                   │
│  EnhancedKG → Clustering Algorithm → Clusters             │
│  Clusters → Community Report Generator → Reports           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              6. GRAPHRAG RETRIEVAL                          │
│  Query → Seed Entity Finding → Graph Traversal             │
│  Graph Traversal + Community Reports → Context            │
└─────────────────────────────────────────────────────────────┘
```

### Adım Adım Açıklama

#### 1. Chunking (Parçalama)

**Amaç**: Uzun metinleri işlenebilir parçalara bölmek

**Stratejiler**:
- **Token-based**: Token sayısına göre (örn: 200 token)
- **Sentence-based**: Cümle sınırlarına göre
- **Semantic**: Anlamsal benzerliğe göre

**Örnek**:
```python
chunker = create_chunker(
    strategy="token_based",
    chunk_size=200,
    overlap_ratio=0.15
)
chunks = chunker.chunk(
    text=text,
    origin_dataset="apple_corpus",
    origin_file="apple_history.txt"
)
```

**Çıktı**: Chunk listesi (her chunk: text, token_count, chunk_id, metadata)

#### 2. Knowledge Graph Extraction

**Amaç**: Chunk'lardan entity ve relation çıkarmak

**Yöntem**: DSPy framework kullanarak LLM ile extraction

**Süreç**:
1. Schema'dan dinamik DSPy signature'ları oluşturulur
2. **Chunk-based Processing**: Her chunk üzerinde bağımsız extraction yapılır
   - Her chunk için entity extraction
   - Her chunk için relation extraction (entity'ler context olarak verilir)
3. Sonuçlar birleştirilir (duplicate entity ve relation'lar otomatik filtrelenir)
4. Self-loop edge'ler filtrelenir (source == target olan edge'ler atlanır)

**Örnek**:
```python
extractor = KGExtractor(schema)
all_entities = set()
all_triples = set()

for chunk in chunks:
    result = extractor.forward(chunk.text)
    entities = json.loads(result.entities)  # [[name, type], ...]
    relations = json.loads(result.relations)  # [[source, relation, target], ...]
    
    # Unique entity ve relation'ları topla
    all_entities.update([(e[0], e[1]) for e in entities])
    all_triples.update([(r[0], r[1], r[2]) for r in relations])
```

**Çıktı**: Unique entity listesi ve relation listesi (triples)

#### 3. Enhanced KG Oluşturma

**Amaç**: Entity ve relation'lardan EnhancedKG yapısı oluşturmak

**Yapı**:
- **KGNode**: id, type, properties, metadata, embedding
- **KGEdge**: source, target, relationship_type, relationship_detail, metadata

**Örnek**:
```python
enhanced_kg = EnhancedKG()

# Node'lar ekle
for entity_name, entity_type in entities_list:
    node = KGNode(id=entity_name, type=entity_type)
    enhanced_kg.add_node(node)

# Edge'ler ekle
for source, relation, target in triples_list:
    edge = KGEdge(
        source=source,
        target=target,
        relationship_type=relation,
        relationship_detail=relation_descriptions.get(relation, f"{source} {relation} {target}"),
        metadata={}
    )
    enhanced_kg.add_edge(edge)
```

**Çıktı**: EnhancedKG objesi (nodes, edges, clusters)

#### 4. Entity Embedding'leri Ekleme

**Amaç**: Node'lara embedding vektörleri eklemek (GraphRAG için gerekli)

**Provider'lar**:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Gemini (text-embedding-004)
- OpenRouter (çeşitli modeller)
- Local (sentence-transformers)

**Örnek**:
```python
embedding_provider = create_embedding_provider(
    provider="openrouter",
    model="openrouter/text-embedding-ada-002"
)
entity_texts = {node_id: node_id for node_id in enhanced_kg.nodes.keys()}
enhanced_kg.add_entity_embeddings(embedding_provider, entity_texts)
```

**Çıktı**: EnhancedKG (her node'da embedding vektörü)

#### 5. Clustering ve Community Reports

**Amaç**: Graph'u cluster'lara bölmek ve her cluster için özet rapor oluşturmak

**Clustering Algoritmaları**:
- **Louvain**: Community detection (python-louvain paketi gerekli)
- **Leiden**: Louvain'in geliştirilmiş versiyonu (leidenalg, python-igraph gerekli)
- **Spectral**: Spectral clustering (scikit-learn gerekli)

**Süreç**:
1. EnhancedKG NetworkX graph formatına çevrilir
2. Seçilen algoritma ile clustering yapılır
3. Cluster'lar EnhancedKG'ye eklenir
4. Her cluster için community report oluşturulur

**Community Reports İçeriği**:
- **Top Actors**: Cluster'daki önemli entity'ler (entity frequency'e göre)
- **Top Relationships**: Cluster'daki önemli ilişkiler (relationship frequency'e göre)
- **Themes**: Cluster'ın temaları (top actors ve relationships'ten çıkarılır)
- **Summary**: Cluster özeti

**Örnek**:
```python
clustering_algorithm = create_clustering_algorithm(algorithm="louvain")
G = nx.Graph()

# EnhancedKG'yi NetworkX'e çevir
for node_id in enhanced_kg.nodes.keys():
    G.add_node(node_id)
for edge in enhanced_kg.edges:
    G.add_edge(edge.source, edge.target)

# Clustering yap
clusters = clustering_algorithm.cluster(G)

# Cluster'ları EnhancedKG'ye ekle
for cluster in clusters:
    kg_cluster = Cluster(
        id=f"cluster_{cluster.cluster_id}",
        node_ids=set(cluster.nodes),
        metadata=cluster.metadata
    )
    enhanced_kg.add_cluster(kg_cluster)

# Community reports oluştur
report_generator = CommunityReportGenerator(enhanced_kg)
reports = report_generator.generate_all_reports()
```

**Çıktı**: Cluster listesi ve Community report listesi

#### 6. GraphRAG Retrieval

**Amaç**: Query'ye göre KG'den ilgili bilgileri retrieve etmek

**Süreç**:
1. **Seed Entity Finding**: Query'yi embed et, KG'deki node embedding'leri ile karşılaştır, en benzer node'ları bul
2. **Graph Traversal**: Seed entity'lerden başlayarak graph'ı traverse et
3. **Subgraph Extraction**: İlgili node ve edge'leri içeren subgraph oluştur
4. **Community Report Integration**: İlgili cluster'ların community report'larını ekle
5. **Context Assembly**: Tüm bilgileri context olarak birleştir

**Örnek**:
```python
retriever = create_graphrag_retriever(
    kg=enhanced_kg,
    embedding_provider=embedding_provider
)
context = retriever.retrieve(
    query="What products does Apple produce?",
    max_hops=2,
    top_k=5
)
```

**Çıktı**: RetrievalContext (entities, relationships, community_reports, chunks)

### Pipeline Çıktıları

Pipeline çalıştıktan sonra şu dosyalar oluşturulur:

- `outputs/{example_name}_schema.json`: Kullanılan schema (Enhanced veya Legacy format)
- `outputs/{example_name}_kg.json`: Oluşturulan Knowledge Graph (EnhancedKG formatında)
- `outputs/{example_name}_community_reports.json`: Community report'lar (clustering yapıldıysa)
- `outputs/{example_name}_summary.json`: Pipeline özeti (chunk sayısı, node sayısı, cluster sayısı, vb.)

**Not**: Dosya isimlendirme formatı sayı başta kullanılır (örn: `1example`, `2example`, `3example`). Pipeline hem eski format (`example1`) hem de yeni format (`1example`) destekler.

---

## Bileşenler ve Metodlar

### Schema Bileşenleri

#### Entity ve Relation Tanımları

```python
# Basit Entity
entity = Entity("Company")

# Basit Relation
relation = Relation(
    name="produces",
    source="Company",
    target="Product",
    description="Şirket ürün üretir"  # Opsiyonel açıklayıcı cümle
)

# DRGSchema
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
```

#### Enhanced Schema Bileşenleri

```python
# EntityType (gelişmiş entity tanımı)
entity_type = EntityType(
    name="Company",
    description="Business organizations and corporations",
    examples=["Apple", "Google", "Microsoft"],
    properties={"industry": "tech"}
)

# RelationGroup (ilişkili relation'ları gruplama)
relation_group = RelationGroup(
    name="production",
    description="How companies create products",
    relations=[
        Relation(
            name="produces",
            src="Company",
            dst="Product",
            description="Relationship type explanation - why this relationship exists",
            detail="Specific detail about why/how entities are connected"
        ),
        Relation(
            name="manufactures",
            src="Company",
            dst="Product",
            description="Manufacturing relationship",
            detail="Companies create products through manufacturing processes"
        )
    ],
    examples=[]  # Opsiyonel: Örnek metinler ve entity/relation'lar
)

# EnhancedDRGSchema
enhanced_schema = EnhancedDRGSchema(
    entity_types=[entity_type],
    relation_groups=[relation_group],
    auto_discovery=True  # Schema'da tanımlı olmayan relation'ları da bul
)
```

### Extraction Metodları

#### `extract_typed(text, schema)`

Metinden typed entity ve relation çıkarır.

**Parametreler**:
- `text` (str): İşlenecek metin
- `schema` (DRGSchema | EnhancedDRGSchema): Şema tanımı

**Döndürür**:
- `Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]`: (entities, triples)
  - entities: `[(entity_name, entity_type), ...]`
  - triples: `[(source, relation, target), ...]`

**Örnek**:
```python
text = "Apple produces iPhone. Tim Cook is the CEO of Apple."
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product"), Entity("Person")],
    relations=[
        Relation("produces", "Company", "Product"),
        Relation("ceo_of", "Person", "Company")
    ]
)

entities, triples = extract_typed(text, schema)
# entities: [("Apple", "Company"), ("iPhone", "Product"), ("Tim Cook", "Person")]
# triples: [("Apple", "produces", "iPhone"), ("Tim Cook", "ceo_of", "Apple")]
```

#### `KGExtractor`

DSPy module olarak extraction yapar.

**Kullanım**:
```python
extractor = KGExtractor(schema)
result = extractor.forward(text)
entities = json.loads(result.entities)
relations = json.loads(result.relations)
```

**Özellikler**:
- Rate limit handling (otomatik retry)
- Exponential backoff
- JSON string parsing (Gemini uyumluluğu için)

### Chunking Metodları

#### `create_chunker(strategy, chunk_size, overlap_ratio, preset)`

Chunker oluşturur.

**Parametreler**:
- `strategy` (str): "token_based" | "sentence_based" | "semantic"
- `chunk_size` (int): Chunk boyutu (token veya cümle sayısı)
- `overlap_ratio` (float): Chunk'lar arası overlap oranı (0.0-1.0)
- `preset` (str): Preset ismi (örn: "graphrag") - preset belirtilirse diğer parametreler override edilir

**Preset'ler**:
- `"graphrag"`: GraphRAG için optimize edilmiş chunking (token_based, chunk_size=200, overlap_ratio=0.15)
- `"medium"`: Orta boyutlu chunk'lar için (token_based, chunk_size=500, overlap_ratio=0.1)

**Örnek**:
```python
# Preset kullanarak
chunker = create_chunker(preset="graphrag")
chunks = chunker.chunk(text, origin_dataset="corpus", origin_file="file.txt")

# Manuel parametrelerle
chunker = create_chunker(
    strategy="token_based",
    chunk_size=200,
    overlap_ratio=0.15
)
chunks = chunker.chunk(text, origin_dataset="corpus", origin_file="file.txt")
```

### Embedding Metodları

#### `create_embedding_provider(provider, model, **kwargs)`

Embedding provider oluşturur.

**Parametreler**:
- `provider` (str): "openai" | "gemini" | "openrouter" | "local"
- `model` (str): Model adı (örn: "text-embedding-3-small")
- `**kwargs`: Provider-specific parametreler

**Örnek**:
```python
# OpenAI
provider = create_embedding_provider(
    provider="openai",
    model="text-embedding-3-small"
)

# OpenRouter
provider = create_embedding_provider(
    provider="openrouter",
    model="openrouter/text-embedding-ada-002"
)

# Local
provider = create_embedding_provider(
    provider="local",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Embedding yap
embedding = provider.embed("Apple")
embeddings = provider.embed_batch(["Apple", "iPhone"])
```

### Knowledge Graph Metodları

#### `EnhancedKG`

Ana Knowledge Graph class'ı.

**Metodlar**:
- `add_node(node)`: Node ekle
- `add_edge(edge)`: Edge ekle
- `add_cluster(cluster)`: Cluster ekle
- `add_entity_embeddings(provider, entity_texts)`: Entity embedding'leri ekle
- `to_dict()`: Dictionary'ye çevir
- `to_json()`: JSON string'e çevir

**Örnek**:
```python
kg = EnhancedKG()

# Node ekle
node = KGNode(id="Apple", type="Company", properties={}, metadata={})
kg.add_node(node)

# Edge ekle
edge = KGEdge(
    source="Apple",
    target="iPhone",
    relationship_type="produces",
    relationship_detail="Apple iPhone üretir",
    metadata={}
)
kg.add_edge(edge)

# Embedding ekle
provider = create_embedding_provider(provider="openai")
entity_texts = {"Apple": "Apple", "iPhone": "iPhone"}
kg.add_entity_embeddings(provider, entity_texts)
```

### Retrieval Metodları

#### `create_graphrag_retriever(kg, embedding_provider)`

GraphRAG retriever oluşturur.

**Parametreler**:
- `kg` (EnhancedKG): Knowledge Graph
- `embedding_provider` (EmbeddingProvider): Embedding provider

**Örnek**:
```python
retriever = create_graphrag_retriever(
    kg=enhanced_kg,
    embedding_provider=embedding_provider
)

context = retriever.retrieve(
    query="What products does Apple produce?",
    max_hops=2,
    top_k=5
)

# Context içeriği:
# - entities: List of entities
# - relationships: List of relationships
# - community_reports: List of community reports
# - chunks: List of relevant chunks
```

### Clustering Metodları

#### `create_clustering_algorithm(algorithm)`

Clustering algorithm oluşturur.

**Parametreler**:
- `algorithm` (str): "louvain" | "leiden" | "spectral"

**Örnek**:
```python
algorithm = create_clustering_algorithm(algorithm="louvain")
G = nx.Graph()  # NetworkX graph
# Graph'u doldur
clusters = algorithm.cluster(G)  # List of node sets
```

---

## Diğer KG Sistemlerinden Farkları

### 1. Declarative vs Imperative

**Geleneksel KG Sistemleri**:
- Imperative yaklaşım: Adım adım manuel işlem
- Kod yazma gereksinimi
- Implementation detaylarına hakim olma zorunluluğu

**DRG**:
- Declarative yaklaşım: Sadece schema tanımlama
- Minimal kod
- Implementation detayları gizli

### 2. Dataset-Agnostic Tasarım

**Geleneksel KG Sistemleri**:
- Genellikle belirli bir domain için optimize edilmiş
- Domain-specific adaptasyon gerektirir

**DRG**:
- Herhangi bir veri kaynağından bağımsız
- Pluggable components ile kolay adaptasyon
- Domain-specific optimizasyonlar core pipeline'ı değiştirmeden eklenebilir

### 3. GraphRAG Desteği

**Geleneksel KG Sistemleri**:
- Genellikle sadece KG oluşturma
- Retrieval için ayrı sistemler gerekir

**DRG**:
- KG oluşturma + GraphRAG retrieval
- End-to-end pipeline
- Community reports ile zengin context

### 4. LLM Entegrasyonu

**Geleneksel KG Sistemleri**:
- Genellikle rule-based veya ML-based extraction
- LLM entegrasyonu manuel

**DRG**:
- DSPy framework ile native LLM entegrasyonu
- Otomatik LLM konfigürasyonu
- Multi-provider desteği (OpenAI, Gemini, Anthropic, vb.)

### 5. Relationship Description Desteği

**Geleneksel KG Sistemleri**:
- Genellikle sadece relation type (örn: "produces")
- Açıklayıcı cümleler yok

**DRG**:
- Relation description desteği
- Her relation için açıklayıcı cümle
- Daha zengin semantic bilgi

### 6. Monolithic-Modular Mimarisi

**Geleneksel KG Sistemleri**:
- Genellikle tamamen modüler (ayrı paketler)
- Veya tamamen monolithic (tek blok)

**DRG**:
- Monolithic-modular hybrid
- Tek deployment unit
- Modüler test edilebilirlik

### 7. Enhanced Schema

**Geleneksel KG Sistemleri**:
- Genellikle basit entity-relation tanımları
- Gruplama ve property desteği sınırlı

**DRG**:
- Enhanced schema (EntityType, RelationGroup, EntityGroup, PropertyGroup)
- Zengin metadata desteği
- Auto-discovery özelliği

---

## Kullanım Örnekleri

### Basit Kullanım

```python
from drg import Entity, Relation, DRGSchema, extract_typed, EnhancedKG, KGNode, KGEdge

# Schema tanımla
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# Metinden çıkarım yap
text = "Apple produces iPhone, iPad, and Mac computers."
entities, triples = extract_typed(text, schema)

# EnhancedKG oluştur
kg = EnhancedKG()
for entity_name, entity_type in entities:
    kg.add_node(KGNode(id=entity_name, type=entity_type))

for source, relation, target in triples:
    kg.add_edge(KGEdge(
        source=source,
        target=target,
        relationship_type=relation,
        relationship_detail=f"{source} {relation} {target}",
        metadata={}
    ))

# JSON'a çevir
print(kg.to_json(indent=2))
```

### Tam GraphRAG Pipeline

```python
from drg.chunking import create_chunker
from drg.embedding import create_embedding_provider
from drg.extract import KGExtractor, _configure_llm_auto
from drg.schema import DRGSchema, Entity, Relation
from drg.graph.kg_core import EnhancedKG, KGNode, KGEdge
from drg.retrieval import create_graphrag_retriever

# 1. Chunking
chunker = create_chunker(strategy="token_based", chunk_size=200)
chunks = chunker.chunk(text, origin_dataset="corpus", origin_file="file.txt")

# 2. LLM konfigürasyonu
_configure_llm_auto()

# 3. KG Extraction
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
extractor = KGExtractor(schema)
result = extractor.forward(text)
entities = json.loads(result.entities)
triples = json.loads(result.relations)

# 4. EnhancedKG oluştur
kg = EnhancedKG()
# ... node ve edge ekleme

# 5. Embedding ekle
provider = create_embedding_provider(provider="openai")
kg.add_entity_embeddings(provider, entity_texts)

# 6. GraphRAG Retrieval
retriever = create_graphrag_retriever(kg=kg, embedding_provider=provider)
context = retriever.retrieve(query="What products does Apple produce?")
```

### JSON Schema ile Kullanım

```python
import json
from drg.schema import DRGSchema, Entity, Relation

# Schema'yı JSON'dan yükle
with open("schema.json", "r") as f:
    schema_data = json.load(f)

entities = [Entity(e["name"]) for e in schema_data["entities"]]
relations = [
    Relation(
        r["name"],
        r["source"],
        r["target"],
        description=r.get("description", "")
    )
    for r in schema_data["relations"]
]

schema = DRGSchema(entities=entities, relations=relations)
```

### Pipeline Example Kullanımı

```bash
# Pipeline'ı çalıştır (sayı başta format)
python examples/graphrag_pipeline_example.py 1
python examples/graphrag_pipeline_example.py 1example
python examples/graphrag_pipeline_example.py example1  # Otomatik 1example'a çevrilir

# Çıktılar:
# - outputs/1example_schema.json
# - outputs/1example_kg.json
# - outputs/1example_community_reports.json (clustering yapıldıysa)
# - outputs/1example_summary.json
```

### Otomatik Schema Generation

Metin verildiğinde, schema yoksa otomatik olarak EnhancedDRGSchema oluşturulur:

```python
from drg.extract import generate_schema_from_text

# Metinden otomatik schema oluştur
text = "Apple Inc. is a technology company..."
schema = generate_schema_from_text(text)

# Schema içeriği:
# - entity_types: Properties ve examples ile zengin entity tanımları
# - relation_groups: Semantic olarak gruplandırılmış relation'lar
# - Her relation için description (bağlantı sebebi) ve detail (bağlantı detayı)
```

### Web API ve Visualization

DRG, FastAPI tabanlı bir web server ve interaktif graph visualization UI sunar:

#### API Server Başlatma

```python
from drg.api import DRGAPIServer
from drg.graph import EnhancedKG

# KG'yi yükle
kg = EnhancedKG.from_json_file("outputs/4example_kg.json")

# API server oluştur ve başlat
server = DRGAPIServer(kg=kg)
server.run(host="0.0.0.0", port=8000)
```

Veya script ile:

```bash
# En son güncellenen KG ile otomatik başlat
python examples/api_server_example.py

# Belirli bir example ile başlat
python examples/api_server_example.py 4example

# Shell script ile (GEMINI_API_KEY otomatik export edilir)
./start_api_server.sh 4example
```

#### API Endpoints

- `GET /` - Web UI (interaktif graph visualization)
- `GET /api/graph` - Full graph data (JSON)
- `GET /api/graph/stats` - Graph statistics
- `GET /api/communities` - All community/cluster data
- `GET /api/visualization/{format}` - Visualization data (cytoscape, vis-network, d3)
- `POST /api/query` - Execute query ve provenance chain al
- `GET /api/provenance/{provenance_id}` - Query provenance chain
- `POST /api/neo4j/sync` - Neo4j'e sync
- `GET /api/neo4j/stats` - Neo4j statistics

#### Web UI Özellikleri

- **Interactive Graph**: Cytoscape.js ile zoom, pan, drag
- **Community Coloring**: Cluster'lara göre node renklendirme
- **Node/Edge Details**: Hover ile detay bilgileri
- **Multiple Layouts**: breadthfirst, concentric, cose, grid, circle
- **Query Interface**: Query girme ve sonuç görüntüleme
- **Provenance Visualization**: Query sonuçlarının provenance chain'i

#### Neo4j Integration

```python
from drg.graph import Neo4jConfig, Neo4jExporter

# Neo4j konfigürasyonu
config = Neo4jConfig(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Exporter oluştur ve sync et
exporter = Neo4jExporter(config)
exporter.sync_kg(enhanced_kg, clear_existing=True)

# Graph statistics
stats = exporter.get_graph_stats()
print(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")
```

---

## Geliştirme ve Katkıda Bulunma

### Geliştirme Ortamı Kurulumu

```bash
# Projeyi klonla
git clone <repository-url>
cd DRG

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate  # Windows

# Projeyi kur
pip install -e .

# Dependencies kur
pip install -r requirements.txt
```

### Test Çalıştırma

```bash
# Tüm testler
pytest tests/

# Belirli bir test
pytest tests/test_basic.py::test_extract_entities_and_relations_with_openai

# API key olmadan yapı testleri
python examples/graphrag_pipeline_example.py example1
```

### Kod Standartları

- **Type Hints**: Tüm fonksiyonlar type hint'li
- **Docstrings**: Google style docstrings
- **Linting**: ruff, mypy, black
- **Interface-First**: Önce interface, sonra implementation

### Yeni Bileşen Ekleme

1. Interface tanımla (`drg/<module>/interface.py`)
2. Implementation yap (`drg/<module>/<provider>.py`)
3. Factory function ekle (`drg/<module>/__init__.py`)
4. Test yaz (`tests/test_<module>.py`)
5. Dokümantasyon güncelle (`docs/<module>.md`)

---

## Sonuç

DRG, modern LLM teknolojilerini kullanarak declarative bir yaklaşımla Knowledge Graph oluşturma ve GraphRAG retrieval yapma imkanı sunan, dataset-agnostic bir Python kütüphanesidir. Proje, research-grade kalitede, community publication-ready bir yapıda tasarlanmıştır.

### Projenin Güçlü Yönleri

1. ✅ **Declarative Yaklaşım**: Minimal kod, maksimum esneklik
2. ✅ **Dataset-Agnostic**: Herhangi bir veri kaynağından bağımsız
3. ✅ **GraphRAG Desteği**: End-to-end GraphRAG pipeline
4. ✅ **Multi-Provider**: OpenAI, Gemini, Anthropic, OpenRouter, vb.
5. ✅ **Enhanced Schema**: Zengin metadata ve açıklama desteği
6. ✅ **Modular Architecture**: Kolay test edilebilirlik ve genişletilebilirlik

### Gelecek Geliştirmeler

- [ ] Neo4j, ArangoDB gibi graph database desteği
- [ ] Daha fazla clustering algoritması
- [ ] Real-time KG update mekanizması
- [ ] Web UI
- [ ] Docker containerization
- [ ] Cloud deployment (AWS, GCP, Azure)

---

**Not**: Bu dokümantasyon, DRG projesinin mevcut durumunu (v0.1.0a0) yansıtmaktadır. API değişiklikleri olabilir.

