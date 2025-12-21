#!/usr/bin/env python3
"""
DRG Workflow: Doküman → Declarative Schema → Knowledge Graph

Bu örnek, doğru workflow'u gösterir:
1. Önce doküman verilir
2. Sonra declarative schema tanımlanır
3. Sonra KG extraction yapılır
"""

import os
import sys
from pathlib import Path

# Proje root'u path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import Entity, Relation, DRGSchema, extract_typed, KG
from drg.chunking import create_chunker


def main():
    print("=" * 70)
    print("📄 DRG Workflow: Doküman → Schema → Knowledge Graph")
    print("=" * 70)
    print()
    
    # API key kontrolü
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY bulunamadı!")
        print("   export GEMINI_API_KEY='your-key'")
        return
    
    os.environ["DRG_MODEL"] = "gemini/gemini-2.0-flash-exp"
    print("✓ API Key ayarlandı")
    print()
    
    # ============================================================
    # ADIM 1: DOKÜMAN VERİLİR
    # ============================================================
    print("=" * 70)
    print("ADIM 1: DOKÜMAN VERİLİR")
    print("=" * 70)
    print()
    
    # Örnek doküman (gerçek kullanımda dosyadan okunur)
    document = """
    Apple Inc. is an American multinational technology company that specializes 
    in consumer electronics, computer software, and online services. The company 
    was founded in April 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
    
    Apple's headquarters is located in Cupertino, California. The company produces 
    various products including the iPhone, iPad, Mac computers, Apple Watch, and 
    AirPods. Tim Cook has been the CEO of Apple since August 2011, succeeding 
    Steve Jobs who resigned due to health issues.
    
    The iPhone was first released in 2007 and revolutionized the smartphone industry. 
    Apple also operates retail stores in 25 countries and has an online store. 
    The company is known for its ecosystem of products and services.
    
    Apple's revenue reached $394.3 billion in 2022. The company employs over 
    164,000 people worldwide and has a market capitalization that has exceeded 
    $3 trillion at various points.
    """
    
    print("📄 Doküman:")
    print(f"   Kelime sayısı: {len(document.split())}")
    print(f"   Karakter sayısı: {len(document)}")
    print(f"   İlk 200 karakter: {document[:200]}...")
    print()
    
    # Opsiyonel: Chunking (uzun dokümanlar için)
    print("📦 Chunking (opsiyonel, uzun dokümanlar için)...")
    chunker = create_chunker(
        strategy="token_based",
        chunk_size=200,  # Küçük chunk size (demo için)
        overlap_ratio=0.15,
    )
    
    chunks = chunker.chunk(
        text=document,
        origin_dataset="demo",
        origin_file="apple_company.txt",
    )
    
    print(f"   {len(chunks)} chunk oluşturuldu")
    print()
    
    # ============================================================
    # ADIM 2: DECLARATIVE SCHEMA TANIMLANIR
    # ============================================================
    print("=" * 70)
    print("ADIM 2: DECLARATIVE SCHEMA TANIMLANIR")
    print("=" * 70)
    print()
    
    print("🔧 Developer sadece NE yapılacağını tanımlar:")
    print("   - Hangi entity tipleri extract edilecek?")
    print("   - Hangi relation tipleri extract edilecek?")
    print()
    
    # Declarative schema tanımlama
    schema = DRGSchema(
        entities=[
            Entity("Company"),
            Entity("Person"),
            Entity("Product"),
            Entity("Location"),
            Entity("Year"),
        ],
        relations=[
            Relation("founded_by", "Company", "Person"),
            Relation("produces", "Company", "Product"),
            Relation("located_in", "Company", "Location"),
            Relation("ceo_of", "Person", "Company"),
            Relation("released_in", "Product", "Year"),
            Relation("employs", "Company", "Person"),
        ],
    )
    
    print("✓ Schema tanımlandı:")
    print(f"   Entities: {[e.name for e in schema.entities]}")
    print(f"   Relations: {len(schema.relations)} adet")
    for rel in schema.relations:
        print(f"     - {rel.name}: {rel.src} → {rel.dst}")
    print()
    print("💡 Not: Developer sadece schema tanımladı, extraction algoritması yazmadı!")
    print()
    
    # ============================================================
    # ADIM 3: KNOWLEDGE GRAPH OLUŞTURULUR
    # ============================================================
    print("=" * 70)
    print("ADIM 3: KNOWLEDGE GRAPH OLUŞTURULUR")
    print("=" * 70)
    print()
    
    print("🔄 DRG otomatik olarak extraction yapıyor...")
    print("   - DSPy signature'ları otomatik oluşturuluyor")
    print("   - LLM ile entity ve relation extraction yapılıyor")
    print("   - Schema'ya göre validation yapılıyor")
    print()
    
    try:
        # Extract entities and relations
        entities, triples = extract_typed(document, schema)
        
        # Remove duplicates
        entities = list(dict.fromkeys(entities))
        triples = list(dict.fromkeys(triples))
        
        print("✅ Extraction tamamlandı!")
        print()
        
        # Sonuçları göster
        print("📊 Sonuçlar:")
        print(f"   Entities: {len(entities)} adet")
        entity_by_type = {}
        for name, etype in entities:
            if etype not in entity_by_type:
                entity_by_type[etype] = []
            entity_by_type[etype].append(name)
        
        for etype, names in entity_by_type.items():
            print(f"     {etype}: {', '.join(names)}")
        
        print()
        print(f"   Relations: {len(triples)} adet")
        for s, r, o in triples[:10]:  # İlk 10'unu göster
            print(f"     {s} --[{r}]--> {o}")
        if len(triples) > 10:
            print(f"     ... ve {len(triples) - 10} adet daha")
        print()
        
        # Knowledge Graph oluştur
        print("📊 Knowledge Graph oluşturuluyor...")
        kg = KG.from_typed(entities, triples)
        
        print(f"   Nodes: {len(kg.nodes)}")
        print(f"   Edges: {len(kg.edges)}")
        print()
        
        # JSON'a kaydet
        output_json = kg.to_json(indent=2)
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/document_to_kg_workflow.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_json)
        
        print(f"💾 Knowledge Graph kaydedildi: {output_file}")
        print()
        
        # Özet
        print("=" * 70)
        print("✅ WORKFLOW TAMAMLANDI")
        print("=" * 70)
        print()
        print("📋 Özet:")
        print(f"   1. Doküman: {len(document.split())} kelime")
        print(f"   2. Schema: {len(schema.entities)} entity tipi, {len(schema.relations)} relation tipi")
        print(f"   3. KG: {len(kg.nodes)} node, {len(kg.edges)} edge")
        print()
        print("💡 Bu workflow tamamen declarative:")
        print("   - Developer sadece doküman ve schema verdi")
        print("   - DRG otomatik olarak extraction yaptı")
        print("   - Manuel algoritma yazılmadı!")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

