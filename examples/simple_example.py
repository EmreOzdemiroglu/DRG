#!/usr/bin/env python3
"""
Basit DRG örneği - API key olmadan da çalışabilir (mock mode)
"""

import os
import sys
from pathlib import Path

# Proje root'u path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import Entity, Relation, DRGSchema, extract_typed, KG

def main():
    print("=" * 60)
    print("🚀 DRG Basit Örnek")
    print("=" * 60)
    
    # API key kontrolü ve environment variable set etme
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = gemini_key or openai_key
    
    if api_key:
        print(f"✓ API key bulundu")
        # Gemini key varsa Gemini model kullan
        if gemini_key:
            model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
        else:
            model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        # Environment variable'ları set et (DSPy otomatik okur)
        os.environ["DRG_MODEL"] = model
        print(f"✓ LLM otomatik konfigüre edilecek: {model}\n")
    else:
        print("⚠️  API key yok - sadece schema ve KG yapısını test ediyoruz\n")
    
    # Schema oluştur
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")]
    )
    print("✓ Schema oluşturuldu")
    print(f"   Entities: {[e.name for e in schema.entities]}")
    print(f"   Relations: {[(r.name, r.src, r.dst) for r in schema.relations]}\n")
    
    # Test metni
    text = "Apple released the iPhone 16 in September 2025. Samsung also produces the Galaxy S24."
    print(f"📄 Test Metni:")
    print(f"   {text}\n")
    
    if not api_key:
        print("⚠️  API key olmadan extraction yapılamaz.")
        print("   Sadece schema ve KG yapısını gösteriyoruz:\n")
        
        # Mock data ile KG oluştur
        entities = [("Apple", "Company"), ("iPhone 16", "Product"), ("Samsung", "Company"), ("Galaxy S24", "Product")]
        triples = [("Apple", "produces", "iPhone 16"), ("Samsung", "produces", "Galaxy S24")]
        
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        
        print("📊 Knowledge Graph (Mock):")
        print(output_json)
        
        # Output'u kaydet
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/simple_example_mock.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
        return
    
    # Extract
    print("🔄 Extraction başlıyor...")
    try:
        entities, triples = extract_typed(text, schema)
        # Remove duplicates
        triples = list(dict.fromkeys(triples))
        
        print(f"\n✅ Sonuçlar:")
        print(f"   {len(entities)} entity bulundu:")
        for name, etype in entities:
            print(f"     - {name} ({etype})")
        
        print(f"\n   {len(triples)} relation bulundu:")
        for s, r, o in triples:
            print(f"     - {s} --[{r}]--> {o}")
        
        # KG oluştur
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        
        print(f"\n📊 Knowledge Graph JSON:")
        print(output_json)
        
        # Output'u kaydet
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/simple_example.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

