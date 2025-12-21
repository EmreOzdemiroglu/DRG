# Rate Limit Handling

## Otomatik Retry Mekanizması

DRG, API rate limit hatalarını otomatik olarak yönetir:

### Özellikler

1. **Otomatik Retry**: Rate limit hatası durumunda otomatik olarak 3 kez deneme yapar
2. **Exponential Backoff**: Her denemede bekleme süresi artar (2s, 4s, 8s)
3. **Akıllı Hata Tespiti**: Sadece rate limit hatalarında retry yapar, diğer hatalarda hemen hata verir

### Kullanım

Rate limit handling otomatik olarak çalışır, ekstra kod gerekmez:

```python
from drg import extract_typed, DRGSchema, Entity, Relation

schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)

# Rate limit durumunda otomatik retry yapılır
entities, triples = extract_typed(text, schema)
```

### Rate Limit Hataları

DRG şu hataları rate limit olarak algılar:
- "rate limit" içeren hatalar
- HTTP 429 status code
- "quota" içeren hatalar

### Bekleme Süreleri

- 1. deneme: 2 saniye bekle
- 2. deneme: 4 saniye bekle
- 3. deneme: 8 saniye bekle

### Rate Limit Aşıldığında

Eğer 3 deneme sonrası hala rate limit hatası alırsanız:

1. **Bekleyin**: Birkaç dakika bekleyip tekrar deneyin
2. **API Key Kontrolü**: API key'inizin limit'ini kontrol edin
3. **Model Değiştirin**: Farklı bir model deneyin (ör. OpenAI)
4. **Batch İşleme**: Metni parçalara bölüp daha küçük batch'ler halinde işleyin

### Örnek Çıktı

```
⚠️  Rate limit hit, retrying in 2.0s (attempt 1/3)
⚠️  Rate limit hit, retrying in 4.0s (attempt 2/3)
✅ Extraction tamamlandı!
```

### Gelişmiş Kullanım

Eğer özel retry ayarları istiyorsanız, `KGExtractor`'ı direkt kullanabilirsiniz:

```python
from drg.extract import KGExtractor

extractor = KGExtractor(schema)
result = extractor.forward(
    text=text,
    max_retries=5,      # Max retry sayısı
    retry_delay=3.0     # İlk retry bekleme süresi (saniye)
)
```

### Gemini API Rate Limits

Gemini API'nin rate limit'leri:
- **Free tier**: 15 requests/minute
- **Paid tier**: Daha yüksek limitler

### Öneriler

1. **Küçük Metinler**: İlk testler için küçük metinler kullanın
2. **Batch İşleme**: Uzun metinleri parçalara bölün
3. **Alternatif Model**: Rate limit'e takılırsanız OpenAI gibi alternatif modeller deneyin
4. **API Key Upgrade**: Daha yüksek limit için paid tier'a geçin

