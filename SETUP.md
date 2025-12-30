# DRG Proje Kurulum ve Çalıştırma Rehberi

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

- **Python**: >= 3.10 (Mevcut: 3.13.2 ✅)
- **pip**: Python paket yöneticisi

### 2. Kurulum

```bash
# Proje dizinine git
cd /Users/helindincel/Desktop/DRG

# Virtual environment oluştur (önerilir)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate  # Windows

# Projeyi kur
pip install -e .

# Veya tüm dependencies ile
pip install -r requirements.txt
pip install -e .
```

### 3. API Key Konfigürasyonu

DRG, LLM API'lerini kullanmak için API key'lere ihtiyaç duyar. En az birini ayarlayın:

```bash
# Gemini API Key (önerilen)
export GEMINI_API_KEY="your-gemini-api-key"

# Veya OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key"

# Model seçimi (opsiyonel, varsayılan: gemini/gemini-2.0-flash-exp)
export DRG_MODEL="gemini/gemini-2.0-flash-exp"
# veya
export DRG_MODEL="openai/gpt-4o-mini"
```

**API Key Olmadan Test:**
- Bazı örnekler API key olmadan da çalışabilir (mock mode)
- Sadece schema ve KG yapısını test eder
- Gerçek extraction için API key gerekli

### 4. Basit Test

```bash
# En basit örnek (API key olmadan da çalışır)
python examples/full_pipeline_example.py 1example

# Tam pipeline örneği (API key gerekli)
python examples/full_pipeline_example.py 1example

# Optimizer örneği (API key gerekli)
python examples/optimizer_demo.py

# MCP API örneği
python examples/mcp_demo.py
```

## 📋 Detaylı Kurulum

### Adım 1: Python Kontrolü

```bash
python --version
# Python 3.10+ olmalı
```

### Adım 2: Dependencies Kurulumu

**Minimum (Sadece Core):**
```bash
pip install dspy>=2.5.0 litellm>=1.0.0
pip install -e .
```

**Tam Kurulum (Tüm Özellikler):**
```bash
pip install -r requirements.txt
pip install -e .
```

**Opsiyonel Paketler:**
- `chromadb`: Vector store için
- `sentence-transformers`: Local embedding için
- `networkx`: Graph processing için
- `python-louvain`, `leidenalg`: Clustering için

### Adım 3: Environment Variables

`.env` dosyası oluşturabilirsiniz (opsiyonel):

```bash
# .env dosyası
GEMINI_API_KEY=your-gemini-api-key
DRG_MODEL=gemini/gemini-2.0-flash-exp
```

Veya direkt export edin:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export DRG_MODEL="gemini/gemini-2.0-flash-exp"
```

### Adım 4: Test

```bash
# Basit test
python examples/full_pipeline_example.py 1example

# Çıktı kontrolü
ls outputs/
```

## 🔧 Sorun Giderme

### Problem: "dspy not found"

**Çözüm:**
```bash
pip install dspy>=2.5.0
```

### Problem: "litellm not found"

**Çözüm:**
```bash
pip install litellm>=1.0.0
```

### Problem: "API key expired"

**Çözüm:**
- Yeni API key alın
- Environment variable'ı güncelleyin
- Terminal'i yeniden başlatın

### Problem: "chromadb not found" (Vector store için)

**Çözüm:**
```bash
pip install chromadb
# Veya mock mode kullanın (API key olmadan)
```

### Problem: "sentence-transformers not found" (Local embedding için)

**Çözüm:**
```bash
pip install sentence-transformers
# Veya API-based embedding kullanın
```

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: API Key Olmadan Test

```bash
python examples/full_pipeline_example.py 1example
# Mock data ile schema ve KG yapısını test eder
```

### Senaryo 2: Basit Extraction (API Key ile)

```bash
export GEMINI_API_KEY="your-key"
python examples/full_pipeline_example.py 1example
# Gerçek extraction yapar
```

### Senaryo 3: Tam Pipeline

```bash
export GEMINI_API_KEY="your-key"
python examples/full_pipeline_example.py 1example
# Chunking + Embedding + KG extraction (+ clustering/raporlar)
```

### Senaryo 4: Optimizer ile İyileştirme

```bash
export GEMINI_API_KEY="your-key"
python examples/optimizer_demo.py
# Iterative learning ile extraction iyileştirme
```

### Senaryo 5: MCP API

```bash
export GEMINI_API_KEY="your-key"
python examples/mcp_demo.py
# Agent interface örneği
```

## 🎯 Hızlı Kontrol Listesi

- [ ] Python 3.10+ yüklü
- [ ] `pip install -e .` çalıştırıldı
- [ ] `dspy` ve `litellm` kuruldu
- [ ] API key ayarlandı (GEMINI_API_KEY veya OPENAI_API_KEY)
- [ ] `python examples/full_pipeline_example.py 1example` çalıştı

## 📚 Daha Fazla Bilgi

- **Dokümantasyon**: `docs/` klasörü
- **Örnekler**: `examples/` klasörü
- **API Referansı**: `README.md`

## 🆘 Yardım

Sorun yaşarsanız:
1. `python --version` ile Python versiyonunu kontrol edin
2. `pip list | grep dspy` ile dspy kurulumunu kontrol edin
3. API key'in doğru ayarlandığını kontrol edin
4. `python examples/full_pipeline_example.py 1example` ile pipeline test yapın

