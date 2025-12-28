# DRG - Declarative Relationship Generation

DRG, metinlerden bilgi grafiği (knowledge graph) çıkarımı yapmak için declarative bir Python kütüphanesidir. DSPy framework'ünü kullanarak, sadece şema tanımlayarak otomatik olarak entity ve relation extraction yapabilirsiniz.

> **⚠️ Note:** This is an alpha version (0.1.0a0). The project is actively under development and may have breaking changes. Use with caution in production environments.

## 🚀 Özellikler

- **Declarative Schema**: Sadece entity tipleri ve ilişkileri tanımlayın, gerisini DRG halletsin
- **DSPy Entegrasyonu**: Modern LLM'lerle çalışan güçlü extraction pipeline
- **Enhanced Schema**: EntityType, RelationGroup, EntityGroup ve PropertyGroup ile zengin şema tanımları
- **Otomatik LLM Konfigürasyonu**: Environment variable'lardan otomatik model ve API key yönetimi
- **CLI Arayüzü**: Komut satırından kolay kullanım
- **Esnek Model Desteği**: OpenAI, Gemini, Anthropic, Perplexity ve Ollama desteği
- **API Key Olmadan Test**: Mock mode ile API key olmadan da şema ve yapı test edilebilir

## 📦 Kurulum

```bash
git clone <repository-url>
cd drg_skeleton

# Geliştirme modunda kurulum
pip install -e .

# Veya direkt kullanım için
pip install .
```

## 🔧 Gereksinimler

- Python >= 3.10
- dspy >= 2.4.0

## ⚙️ Konfigürasyon

DRG, environment variable'lar üzerinden otomatik konfigürasyon yapar:

```bash
# Model seçimi (varsayılan: openai/gpt-4o-mini)
export DRG_MODEL="openai/gpt-4o-mini"
# veya
export DRG_MODEL="gemini/gemini-2.0-flash-exp"
export DRG_MODEL="ollama_chat/llama3"  # Local model

# API Key'ler (model tipine göre)
export OPENAI_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export PERPLEXITY_API_KEY="your-key-here"

# Opsiyonel ayarlar
export DRG_BASE_URL="http://localhost:11434"  # Ollama için
export DRG_TEMPERATURE="0.0"
```

## 📖 Kullanım

### Basit Kullanım

```python
from drg import Entity, Relation, DRGSchema, extract_typed, KG

# Şema tanımla
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# Metinden çıkarım yap
text = "Apple released the iPhone 16 in September 2025."
entities, triples = extract_typed(text, schema)

# Knowledge Graph oluştur
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

### Enhanced Schema ile Kullanım

```python
from drg import (
    EntityType,
    RelationGroup,
    Relation,
    EnhancedDRGSchema,
    extract_typed,
    KG,
)

# Gelişmiş şema tanımla
schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(
            name="Company",
            description="Business organizations that produce products",
            examples=["Apple", "Google", "Microsoft"],
            properties={"industry": "tech"}
        ),
        EntityType(
            name="Product",
            description="Goods produced by companies",
            examples=["iPhone", "Android", "Windows"]
        )
    ],
    relation_groups=[
        RelationGroup(
            name="production",
            description="How companies create products",
            relations=[
                Relation("produces", "Company", "Product"),
                Relation("manufactures", "Company", "Product")
            ]
        )
    ],
    auto_discovery=True
)

# Çıkarım yap
text = "Apple produces iPhones. Google develops Android."
entities, triples = extract_typed(text, schema)
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

## 🖥️ CLI Kullanımı

```bash
# Dosyadan çıkarım
drg extract input.txt -o output.json

# Standart girişten
echo "Apple released iPhone 16" | drg extract - -o output.json

# Özel model ile
drg extract input.txt -o output.json --model "gemini/gemini-2.0-flash-exp"

# Ollama ile (local)
drg extract input.txt -o output.json \
  --model "ollama_chat/llama3" \
  --base-url "http://localhost:11434"

# Özel şema ile (gelecekte)
drg extract input.txt -o output.json --schema custom_schema.json
```

## 📚 API Referansı

### Schema Sınıfları

#### `DRGSchema` (Legacy)
Basit entity ve relation tanımları için.

```python
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
```

#### `EnhancedDRGSchema`
Gelişmiş şema tanımları için.

```python
schema = EnhancedDRGSchema(
    entity_types=[...],
    relation_groups=[...],
    entity_groups=[...],  # Opsiyonel
    property_groups=[...],  # Opsiyonel
    auto_discovery=False
)
```

### Extraction Fonksiyonları

#### `extract_typed(text, schema)`
Metinden entity ve relation çıkarır.

**Parametreler:**
- `text` (str): İşlenecek metin
- `schema` (DRGSchema | EnhancedDRGSchema): Şema tanımı

**Döndürür:**
- `Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]`: (entities, triples)
  - entities: `[(entity_name, entity_type), ...]`
  - triples: `[(source, relation, target), ...]`

#### `extract_triples(text, schema)`
Sadece relation'ları çıkarır (geriye dönük uyumluluk için).

### Graph Sınıfı

#### `KG`
Knowledge Graph temsil sınıfı.

```python
# Typed entities ile oluştur
kg = KG.from_typed(entities, triples)

# Sadece triples ile oluştur
kg = KG.from_triples(triples)

# JSON'a çevir
json_str = kg.to_json(indent=2)
```

## 📁 Proje Yapısı

```
drg_skeleton/
├── drg/
│   ├── __init__.py      # Ana modül export'ları
│   ├── schema.py        # Şema tanımları
│   ├── extract.py       # DSPy extraction logic
│   ├── graph.py         # Knowledge Graph sınıfı
│   └── cli.py           # Komut satırı arayüzü
├── examples/
│   ├── graphrag_pipeline_example.py  # Ana GraphRAG pipeline örneği
│   ├── mcp_demo.py                   # MCP API demo
│   └── optimizer_demo.py             # Optimizer demo
├── tests/
│   └── test_basic.py    # Temel testler
├── outputs/             # Çıktı dosyaları
├── pyproject.toml        # Proje konfigürasyonu
└── README.md
```

## 🧪 Test

```bash
# Testleri çalıştır (API key gerekli)
pytest tests/

# API key olmadan sadece yapı testleri
python examples/graphrag_pipeline_example.py example1
```

## 💡 Örnekler

Detaylı örnekler için `examples/` dizinindeki dosyalara bakın:

- `graphrag_pipeline_example.py`: Tam GraphRAG pipeline (chunking, KG extraction, embedding, retrieval)
- `mcp_demo.py`: MCP API interface demo
- `optimizer_demo.py`: DSPy optimizer demo

## 🔍 Desteklenen Modeller

DRG, DSPy üzerinden aşağıdaki model türlerini destekler:

- **OpenAI**: `openai/gpt-4o-mini`, `openai/gpt-4`, vb.
- **Google Gemini**: `gemini/gemini-2.0-flash-exp`, vb.
- **Anthropic**: `anthropic/claude-3-5-sonnet`, vb.
- **Perplexity**: `perplexity/llama-3.1-sonar-large-128k-online`, vb.
- **Ollama** (Local): `ollama_chat/llama3`, `ollama_chat/mistral`, vb.

Model seçimi `DRG_MODEL` environment variable'ı ile yapılır.

## 🛠️ Geliştirme

```bash
# Geliştirme ortamını kur (tüm optional dependencies ile)
pip install -e ".[dev,all]"

# Testleri çalıştır
pytest

# Linting ve type checking
# (projeye göre eklenebilir: ruff, mypy, black)
```

### Optional Dependencies

DRG, modüler bir bağımlılık yapısı kullanır:

- **Core**: `dspy`, `pydantic` (her zaman gerekli)
- **Graph Persistence**: `neo4j` (Neo4j export için)
- **API Server**: `fastapi`, `uvicorn` (REST API için)
- **Embedding Providers**: `openai`, `google-generativeai`, `sentence-transformers`
- **Vector Stores**: `chromadb`, `qdrant-client`, `pinecone-client`, `faiss-cpu`
- **Clustering**: `python-louvain`, `leidenalg`, `scikit-learn`
- **Graph Processing**: `networkx`

Sadece kullandığınız özellikler için ilgili dependencies'i yükleyin.

## 📝 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakın.



**Not**: Bu proje alpha aşamasındadır (v0.1.0a0). API değişiklikleri olabilir.
