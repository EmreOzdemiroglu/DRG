# API Server Yeniden Başlatma

Eğer tarayıcıda eski KG görüyorsanız:

1. **Terminal'de çalışan server'ı durdurun** (Ctrl+C)

2. **Yeni server'ı başlatın:**
```bash
export GEMINI_API_KEY="REDACTED_GOOGLE_API_KEY"
python3 examples/api_server_example.py
```

3. **Tarayıcıyı HARD REFRESH yapın:**
   - **Mac:** Cmd + Shift + R
   - **Windows/Linux:** Ctrl + Shift + F5
   
   Bu, cache'i temizleyerek yeni KG'ı yükler.

4. **Veya tarayıcı cache'ini temizleyin:**
   - Chrome: Settings > Privacy > Clear browsing data > Cached images and files

**Not:** Server otomatik olarak en son güncellenen KG dosyasını yükler (outputs/ klasöründeki en yeni *example*_kg.json dosyası).

