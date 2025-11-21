#!/usr/bin/env python3
"""DRG Test Demo - API key veya Ollama ile test edebilirsiniz"""

import os
from drg import Entity, Relation, DRGSchema, extract_typed, KG
from drg.optimize import refine_triples

def test_with_api_key():
    """Cloud model ile test (API key gerekli)"""
    print("=" * 60)
    print("TEST: Cloud Model (API Key gerekli)")
    print("=" * 60)
    
    # Önce Gemini, sonra OpenAI/Anthropic kontrol et
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ API key bulunamadı. GEMINI_API_KEY, OPENAI_API_KEY veya ANTHROPIC_API_KEY set edin.")
        return False
    
    try:
        # Model seç - Gemini varsa onu kullan
        if os.getenv("GEMINI_API_KEY"):
            model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
        else:
            model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        print(f"📝 Model: {model}")
        
        # Environment variable'ları set et (DSPy otomatik okur)
        os.environ["DRG_MODEL"] = model
        print("✓ LLM otomatik konfigüre edilecek")
        
        # Schema oluştur
        schema = DRGSchema(
            entities=[Entity("Company"), Entity("Product")],
            relations=[Relation("produces", "Company", "Product")]
        )
        print("✓ Schema oluşturuldu")
        
        # Test metni
        text = "Apple released the iPhone 16 in September 2025. Samsung also produces the Galaxy S24."
        print(f"\n📄 Metin: {text}\n")
        
        # Extract
        print("🔄 Extraction başlıyor...")
        entities, triples = extract_typed(text, schema)
        triples = refine_triples(triples)
        
        print(f"✓ {len(entities)} entity bulundu")
        print(f"✓ {len(triples)} relation bulundu\n")
        
        # KG oluştur
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        print("📊 Knowledge Graph:")
        print(output_json)
        
        # Output'u dosyaya kaydet
        import os
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/test_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_ollama():
    """Local Ollama model ile test (API key gerekmez)"""
    print("\n" + "=" * 60)
    print("TEST: Local Ollama Model (API key gerekmez)")
    print("=" * 60)
    
    try:
        # Ollama kontrolü
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                print("❌ Ollama çalışmıyor. 'ollama serve' komutu ile başlatın.")
                return False
        except:
            print("❌ Ollama çalışmıyor. 'ollama serve' komutu ile başlatın.")
            return False
        
        # Model seç (Ollama'da yüklü bir model olmalı)
        model = "ollama_chat/llama3"  # veya başka bir model
        print(f"📝 Model: {model}")
        
        # Environment variable'ları set et (DSPy otomatik okur)
        os.environ["DRG_MODEL"] = model
        os.environ["DRG_BASE_URL"] = "http://localhost:11434"
        print("✓ LLM otomatik konfigüre edilecek")
        
        # Schema oluştur
        schema = DRGSchema(
            entities=[Entity("Company"), Entity("Product")],
            relations=[Relation("produces", "Company", "Product")]
        )
        print("✓ Schema oluşturuldu")
        
        # Test metni
        text = "Apple released the iPhone 16 in September 2025."
        print(f"\n📄 Metin: {text}\n")
        
        # Extract
        print("🔄 Extraction başlıyor...")
        entities, triples = extract_typed(text, schema)
        triples = refine_triples(triples)
        
        print(f"✓ {len(entities)} entity bulundu")
        print(f"✓ {len(triples)} relation bulundu\n")
        
        # KG oluştur
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        print("📊 Knowledge Graph:")
        print(output_json)
        
        # Output'u dosyaya kaydet
        import os
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/test_output_ollama.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
        
        return True
        
    except ImportError:
        print("❌ 'requests' modülü gerekli. 'uv pip install requests' ile yükleyin.")
        return False
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 DRG Test Demo\n")
    
    # Önce API key ile dene
    if test_with_api_key():
        print("\n✅ Cloud model testi başarılı!")
    else:
        print("\n⚠️  Cloud model testi atlandı (API key yok)")
    
    # Sonra Ollama ile dene
    if test_with_ollama():
        print("\n✅ Ollama testi başarılı!")
    else:
        print("\n⚠️  Ollama testi atlandı (Ollama çalışmıyor)")
    
    print("\n" + "=" * 60)
    print("💡 İpucu:")
    print("  - Gemini ile test: export GEMINI_API_KEY='your-key'")
    print("  - OpenAI ile test: export OPENAI_API_KEY='your-key'")
    print("  - Ollama ile test: ollama serve (başka terminalde)")
    print("=" * 60 + "\n")

