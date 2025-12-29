# DRG API Server Kullanım Kılavuzu

## Hızlı Başlangıç

### 1. API Key Ayarlama

**Güvenlik Uyarısı**: API key'i kodda hardcode etmeyin! Environment variable kullanın.

#### Yöntem 1: Environment Variable (Önerilen)

**Gemini kullanarak:**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
python examples/api_server_example.py
```

**OpenAI kullanarak (alternatif):**
```bash
export OPENAI_API_KEY="sk-or-v1-..."
python examples/api_server_example.py
```

#### Yöntem 2: Script ile (Kolay Yol)

```bash
# Varsayılan example (1example)
./start_api_server.sh

# Belirli example ile
./start_api_server.sh 3     # 3example
./start_api_server.sh 4     # 4example

# Environment variable ile
DRG_EXAMPLE=3example ./start_api_server.sh
```

**Not**: `start_api_server.sh` script'i API key ve example seçimini otomatikleştirir.

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. API Server'ı Başlatma

**Varsayılan (1example):**
```bash
python examples/api_server_example.py
```

**Belirli bir example ile:**
```bash
# Command line argument ile
python examples/api_server_example.py 3        # 3example
python examples/api_server_example.py 1        # 1example
python examples/api_server_example.py 4example # 4example

# Environment variable ile
export DRG_EXAMPLE=3example
python examples/api_server_example.py

# Script ile (kolay yol)
./start_api_server.sh 3    # 3example
./start_api_server.sh 1    # 1example
```

### 4. Web UI'ya Erişim

- **Web UI**: http://localhost:8000
- **API Dokümantasyonu**: http://localhost:8000/docs
- **Graph API**: http://localhost:8000/api/graph
- **Communities API**: http://localhost:8000/api/communities

## API Endpoints

### Graph Endpoints

- `GET /api/graph` - Tüm graph verilerini getir
- `GET /api/graph/stats` - Graph istatistiklerini getir

### Community Endpoints

- `GET /api/communities` - Tüm community/cluster verilerini getir
- `GET /api/communities/{cluster_id}` - Belirli bir community report'unu getir

### Visualization Endpoints

- `GET /api/visualization/{format}` - Graph visualization verilerini getir
  - Format: `cytoscape`, `vis-network`, `d3`
- `GET /api/visualization/communities/{format}` - Community renk kodlamalı visualization

### Query Endpoints

> Not: Bu repo “query/retrieval serving” hedeflemediği için query endpoint'leri devre dışıdır (KG extraction + görselleştirme odaklı).

### Neo4j Endpoints (Opsiyonel)

- `POST /api/neo4j/sync` - Knowledge graph'ı Neo4j'e senkronize et
- `GET /api/neo4j/stats` - Neo4j graph istatistiklerini getir

## Neo4j Kullanımı (Opsiyonel)

Neo4j kullanmak için environment variable'ları ayarlayın:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

## Özellikler

- ✅ Full graph visualization (Cytoscape.js)
- ✅ Community/cluster visualization with color coding
- ✅ Query provenance chains (query → chunks → community → summary → answer)
- ✅ Semantic similarity weights
- ✅ Interactive graph exploration
- ✅ Neo4j persistence (optional)
- ✅ RESTful API with OpenAPI documentation

## Güvenlik Notları

⚠️ **ÖNEMLİ**: 
- API key'lerinizi kodda hardcode etmeyin
- `start_api_server.sh` gibi key içeren dosyaları git'e commit etmeyin
- Production ortamında environment variable veya secrets management kullanın
- `.env` dosyalarını `.gitignore`'a ekleyin

## Sorun Giderme

### API Key Hatası

Eğer embedding provider oluşturulamazsa, API key'in doğru ayarlandığından emin olun:

```bash
# Gemini için
echo $GEMINI_API_KEY

# OpenAI için (alternatif)
echo $OPENAI_API_KEY
```

### Port Zaten Kullanımda

Eğer port 8000 zaten kullanımdaysa, farklı bir port belirtin:

```python
server.run(host="0.0.0.0", port=8080)
```

### Neo4j Bağlantı Hatası

Neo4j kullanmıyorsanız, Neo4j endpoint'leri çalışmayacaktır. Bu normaldir ve graph visualization için gerekli değildir.
