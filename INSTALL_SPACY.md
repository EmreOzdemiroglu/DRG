# spaCy Model Kurulumu

Coreference resolution özelliğini kullanmak için spaCy English model'ini kurmanız gerekiyor.

## Kurulum

### 1. Model'i İndirin

```bash
python3 -m spacy download en_core_web_sm
```

**Alternatif:** Daha büyük model (daha iyi performans):

```bash
python3 -m spacy download en_core_web_md
```

### 2. Kurulumu Kontrol Edin

```bash
python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ Model başarıyla yüklendi!')"
```

### 3. Coreference Resolution Testi

```bash
python3 -c "from drg.coreference_resolution import CoreferenceResolver; r = CoreferenceResolver(use_nlp=True); print('✅ CoreferenceResolver hazır!' if r.nlp else '❌ Model yüklü değil')"
```

## Notlar

- **en_core_web_sm**: Küçük model (~12 MB) - hızlı, temel özellikler
- **en_core_web_md**: Orta model (~40 MB) - daha iyi performans, word vectors içerir
- Model yüklenmezse, coreference resolution otomatik olarak heuristics moduna geçer

## Kullanım

Model kurulduktan sonra, coreference resolution otomatik olarak NLP kullanacaktır:

```python
from drg.extract import extract_typed
from drg.schema import EnhancedDRGSchema

entities, relations = extract_typed(
    text="Elon Musk founded Tesla. He is the CEO.",
    schema=schema,
    enable_coreference_resolution=True,  # NLP kullanır
    enable_entity_resolution=True
)
```

