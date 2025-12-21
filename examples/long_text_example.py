#!/usr/bin/env python3
"""
Uzun metin ile DRG extraction örneği
"""

import os
import sys
from pathlib import Path

# Proje root'u path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import Entity, Relation, DRGSchema, extract_typed, KG

# Uzun örnek metin (Wikipedia'dan uyarlanmış)
LONG_TEXT = """
Apple Inc. is an American multinational technology company that specializes in consumer electronics, 
computer software, and online services. Apple is the world's largest technology company by revenue 
and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's 
fourth-largest PC vendor by unit sales and fourth-largest smartphone manufacturer.

The company was founded in April 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and 
sell Wozniak's Apple I personal computer. It was incorporated as Apple Computer, Inc. in January 
1977, and sales of its computers, including the Apple II, grew significantly. Throughout the 1980s, 
Apple developed a reputation for innovation, particularly after the 1984 launch of the Macintosh, 
which introduced the graphical user interface to a wide audience.

In 1997, Apple was on the verge of bankruptcy. Jobs returned to the company as CEO and began a 
turnaround by introducing a new line of products, including the iMac in 1998. Under Jobs' leadership, 
Apple introduced several successful products, including the iPod in 2001, the iPhone in 2007, and the 
iPad in 2010. These products revolutionized their respective markets and established Apple as one 
of the world's most valuable companies.

Tim Cook became CEO in August 2011, following Jobs' resignation due to health issues. Under Cook's 
leadership, Apple has continued to expand its product line, introducing new iPhone models, the Apple 
Watch in 2015, AirPods in 2016, and various services including Apple Music, Apple TV+, and Apple Arcade.

Apple's headquarters, Apple Park, is located in Cupertino, California. The company operates 
retail stores in 25 countries and has an online store. Apple's products are sold worldwide through 
its retail stores, online stores, and third-party retailers. The company is known for its 
ecosystem of products and services, which work seamlessly together.

Apple has been involved in various legal disputes throughout its history, including patent 
litigation with competitors like Samsung and Microsoft. The company has also faced criticism 
regarding its labor practices, environmental impact, and business practices.

Despite these challenges, Apple has maintained its position as a leader in the technology industry, 
with a strong focus on design, user experience, and innovation. The company's products are widely 
recognized for their quality, design, and integration with Apple's ecosystem of services.

Apple's revenue has grown significantly over the years, reaching $394.3 billion in 2022. The company 
employs over 164,000 people worldwide and has a market capitalization that has exceeded $3 trillion 
at various points. Apple's success is attributed to its ability to create products that consumers 
love, its strong brand loyalty, and its ecosystem of interconnected products and services.

The company continues to invest heavily in research and development, with R&D spending reaching 
$26.2 billion in 2022. Apple is also committed to environmental sustainability, with a goal of 
becoming carbon neutral across its entire business by 2030. The company has made significant 
progress in using renewable energy for its operations and has eliminated many harmful chemicals 
from its manufacturing processes.

Apple's influence extends beyond technology, with the company playing a significant role in popular 
culture, design, and business strategy. The company's marketing campaigns, product launches, and 
corporate culture have been widely studied and emulated by other companies in various industries.
"""


def main():
    print("=" * 70)
    print("🚀 DRG Uzun Metin Extraction Örneği")
    print("=" * 70)
    print()
    
    # API key kontrolü
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    api_key = gemini_key or openai_key
    
    if not api_key:
        print("❌ API key bulunamadı!")
        print("   Lütfen şu komutu çalıştırın:")
        print("   export GEMINI_API_KEY='your-api-key'")
        return
    
    # Model seçimi
    if gemini_key:
        model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
        os.environ["DRG_MODEL"] = model
        print(f"✓ Gemini API Key bulundu")
        print(f"✓ Model: {model}\n")
    else:
        model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        os.environ["DRG_MODEL"] = model
        print(f"✓ OpenAI API Key bulundu")
        print(f"✓ Model: {model}\n")
    
    # Schema oluştur
    schema = DRGSchema(
        entities=[
            Entity("Company"),
            Entity("Person"),
            Entity("Product"),
            Entity("Location"),
            Entity("Event"),
            Entity("Year"),
        ],
        relations=[
            Relation("founded_by", "Company", "Person"),
            Relation("produces", "Company", "Product"),
            Relation("located_in", "Company", "Location"),
            Relation("led_by", "Company", "Person"),
            Relation("released_in", "Product", "Year"),
            Relation("introduced", "Company", "Product"),
            Relation("acquired", "Company", "Company"),
        ]
    )
    
    print("✓ Schema oluşturuldu")
    print(f"   Entities: {[e.name for e in schema.entities]}")
    print(f"   Relations: {len(schema.relations)} relation tipi\n")
    
    # Metin bilgisi
    word_count = len(LONG_TEXT.split())
    char_count = len(LONG_TEXT)
    print(f"📄 Metin Bilgisi:")
    print(f"   Kelime sayısı: {word_count:,}")
    print(f"   Karakter sayısı: {char_count:,}")
    print(f"   Paragraf sayısı: {len([p for p in LONG_TEXT.split('\\n\\n') if p.strip()])}\n")
    
    # Extraction
    print("🔄 Extraction başlıyor...")
    print("   (Bu işlem birkaç dakika sürebilir)")
    print("   ⚠️  Rate limit durumunda otomatik retry yapılacak (max 3 deneme)\n")
    
    try:
        # Rate limit handling otomatik olarak extract_typed içinde yapılıyor
        entities, triples = extract_typed(LONG_TEXT, schema)
        
        # Duplicate'leri temizle
        entities = list(dict.fromkeys(entities))
        triples = list(dict.fromkeys(triples))
        
        print("✅ Extraction tamamlandı!\n")
        
        # Sonuçları göster
        print("=" * 70)
        print("📊 EXTRACTION SONUÇLARI")
        print("=" * 70)
        print()
        
        print(f"📌 Entities ({len(entities)} adet):")
        entity_by_type = {}
        for name, etype in entities:
            if etype not in entity_by_type:
                entity_by_type[etype] = []
            entity_by_type[etype].append(name)
        
        for etype, names in entity_by_type.items():
            print(f"   {etype}: {len(names)} adet")
            for name in names[:5]:  # İlk 5'ini göster
                print(f"     - {name}")
            if len(names) > 5:
                print(f"     ... ve {len(names) - 5} adet daha")
        print()
        
        print(f"🔗 Relations ({len(triples)} adet):")
        relation_by_type = {}
        for s, r, o in triples:
            if r not in relation_by_type:
                relation_by_type[r] = []
            relation_by_type[r].append((s, o))
        
        for rtype, pairs in relation_by_type.items():
            print(f"   {rtype}: {len(pairs)} adet")
            for s, o in pairs[:3]:  # İlk 3'ünü göster
                print(f"     - {s} --[{rtype}]--> {o}")
            if len(pairs) > 3:
                print(f"     ... ve {len(pairs) - 3} adet daha")
        print()
        
        # Knowledge Graph oluştur
        print("📊 Knowledge Graph oluşturuluyor...")
        kg = KG.from_typed(entities, triples)
        
        # İstatistikler
        print(f"   Nodes: {len(kg.nodes)}")
        print(f"   Edges: {len(kg.edges)}")
        print()
        
        # JSON'a kaydet
        output_json = kg.to_json(indent=2)
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/long_text_example.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        
        print(f"💾 Knowledge Graph kaydedildi: {output_file}")
        print(f"   Dosya boyutu: {len(output_json):,} bytes")
        print()
        
        # Özet
        print("=" * 70)
        print("📈 ÖZET")
        print("=" * 70)
        print(f"   Metin: {word_count:,} kelime, {char_count:,} karakter")
        print(f"   Entities: {len(entities)} adet ({len(entity_by_type)} tip)")
        print(f"   Relations: {len(triples)} adet ({len(relation_by_type)} tip)")
        print(f"   Knowledge Graph: {len(kg.nodes)} node, {len(kg.edges)} edge")
        print()
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

