# 🚀 DRG Hızlı Başlangıç

## Projeyi Çalıştırmak İçin Gerekenler

### ✅ Minimum Gereksinimler

1. **Python 3.10+** (Mevcut: 3.13.2 ✅)
2. **pip** (Python paket yöneticisi)
3. **API Key** (opsiyonel - mock mode ile test edilebilir)

### 📦 Hızlı Kurulum (3 Adım)

#### Adım 1: Dependencies Kurulumu

```bash
cd /Users/helindincel/Desktop/DRG

# Core dependencies
pip install dspy>=2.4.0 litellm>=1.0.0

# Projeyi kur
pip install -e .
```

#### Adım 2: API Key Ayarlama (Opsiyonel)

```bash
# Gemini API Key (önerilen)
export GEMINI_API_KEY="your-gemini-api-key"
export DRG_MODEL="gemini/gemini-2.0-flash-exp"

# Veya OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key"
export DRG_MODEL="openai/gpt-4o-mini"
```

**Not:** API key olmadan da test edebilirsiniz (mock mode)

#### Adım 3: Test Çalıştırma

```bash
# En basit örnek (API key olmadan da çalışır)
python examples/simple_example.py

# Veya otomatik script
./quick_start.sh
```

### 🎯 Hızlı Test Senaryoları

#### Senaryo 1: API Key Olmadan Test
```bash
python examples/simple_example.py
# Mock data ile schema ve KG yapısını test eder
```

#### Senaryo 2: API Key ile Gerçek Extraction
```bash
export GEMINI_API_KEY="your-key"
python examples/simple_example.py
# Gerçek LLM extraction yapar
```

#### Senaryo 3: Tam Pipeline
```bash
export GEMINI_API_KEY="your-key"
python examples/pipeline_example.py
# Chunking + Embedding + RAG + KG extraction
```

### 📋 Kontrol Listesi

Çalıştırmadan önce kontrol edin:

- [ ] Python 3.10+ yüklü (`python --version`)
- [ ] `dspy` kurulu (`pip list | grep dspy`)
- [ ] `litellm` kurulu (`pip list | grep litellm`)
- [ ] Proje kurulu (`pip install -e .`)
- [ ] API key ayarlandı (opsiyonel)

### 🔧 Sorun Giderme

**Problem: "dspy not found"**
```bash
pip install dspy>=2.4.0
```

**Problem: "litellm not found"**
```bash
pip install litellm>=1.0.0
```

**Problem: "API key expired"**
- Yeni API key alın
- `export GEMINI_API_KEY="new-key"` ile güncelleyin

**Problem: "Permission denied"**
```bash
# Virtual environment kullanın
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 📚 Örnekler

Tüm örnekler `examples/` klasöründe:

- `simple_example.py` - En basit kullanım
- `pipeline_example.py` - Tam pipeline
- `optimizer_demo.py` - Optimizer ile iyileştirme
- `mcp_demo.py` - MCP API örneği

### 🎉 Başarılı Kurulum

Kurulum başarılıysa şunu görmelisiniz:

```
🚀 DRG Basit Örnek
============================================================
✓ Schema oluşturuldu
📄 Test Metni: ...
✅ Sonuçlar:
   X entity bulundu
   Y relation bulundu
📊 Knowledge Graph JSON: ...
💾 Output kaydedildi: outputs/simple_example.json
```

### 📖 Daha Fazla Bilgi

- **Detaylı Kurulum**: `SETUP.md`
- **Dokümantasyon**: `docs/` klasörü
- **API Referansı**: `README.md`

