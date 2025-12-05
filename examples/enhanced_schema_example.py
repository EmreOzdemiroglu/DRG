#!/usr/bin/env python3
"""
Enhanced DRG Schema Örneği - Declarative yapı ile.
Koşum: `uv run python examples/enhanced_schema_example.py`
"""

import os
import sys
from pathlib import Path

# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import (
    EntityType,
    EntityGroup,
    PropertyGroup,
    Relation,
    RelationGroup,
    EnhancedDRGSchema,
    extract_typed,
    KG,
)


OUTPUT_DIR = Path("outputs")


def main():
    print("=" * 60)
    print("🚀 Enhanced DRG Schema Örneği")
    print("=" * 60)
    
    # Enhanced Schema oluştur
    business_schema = EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Company",
                description="Business organizations that produce products or provide services",
                examples=["Apple", "Google", "Microsoft", "Samsung"],
                properties={"industry": "tech", "type": "corporation"},
            ),
            EntityType(
                name="Product",
                description="Tangible or intangible goods produced by companies",
                examples=["iPhone", "Android", "Windows", "Galaxy"],
                properties={"category": "technology"},
            ),
            EntityType(
                name="Person",
                description="Individuals associated with companies or products",
                examples=["Tim Cook", "Sundar Pichai", "Satya Nadella"],
                properties={"role": "executive"},
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="production",
                description="How companies create and distribute products",
                relations=[
                    Relation("produces", "Company", "Product"),
                    Relation("manufactures", "Company", "Product"),
                    Relation("develops", "Company", "Product"),
                ],
                examples=[
                    {
                        "text": "Apple produces iPhones",
                        "entities": [("Apple", "Company"), ("iPhone", "Product")],
                        "relations": [("Apple", "produces", "iPhone")],
                    }
                ],
            ),
            RelationGroup(
                name="employment",
                description="People working for companies",
                relations=[
                    Relation("employs", "Company", "Person"),
                    Relation("CEO_of", "Person", "Company"),
                    Relation("founder_of", "Person", "Company"),
                ],
            ),
        ],
        auto_discovery=True,  # Automatically find new relation patterns
    )
    
    print("✓ Enhanced Schema oluşturuldu")
    print(f"\n📋 Schema Özeti:")
    summary = business_schema.get_schema_summary()
    print(f"   - {len(summary['entity_types'])} Entity Type")
    print(f"   - {len(summary['relation_groups'])} Relation Group")
    print(f"   - Auto Discovery: {summary['auto_discovery']}")
    
    print(f"\n📊 Entity Types:")
    for et in business_schema.entity_types:
        print(f"   - {et.name}: {et.description}")
        if et.examples:
            print(f"     Örnekler: {', '.join(et.examples[:3])}")
    
    print(f"\n🔗 Relation Groups:")
    for rg in business_schema.relation_groups:
        print(f"   - {rg.name}: {rg.description}")
        for rel in rg.relations:
            print(f"     • {rel.name}: {rel.src} -> {rel.dst}")
    
    # Test metni
    text = "Apple produces the iPhone 16. Tim Cook is the CEO of Apple. Google develops Android."
    print(f"\n📄 Test Metni:")
    print(f"   {text}\n")
    
    # API key kontrolü
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("⚠️  API key yok - sadece schema yapısını gösteriyoruz\n")
        
        # Mock data ile KG oluştur
        entities = [
            ("Apple", "Company"),
            ("iPhone 16", "Product"),
            ("Tim Cook", "Person"),
            ("Google", "Company"),
            ("Android", "Product"),
        ]
        triples = [
            ("Apple", "produces", "iPhone 16"),
            ("Tim Cook", "CEO_of", "Apple"),
            ("Google", "develops", "Android"),
        ]
        
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        
        print("📊 Knowledge Graph (Mock):")
        print(output_json)
        
        # Output'u kaydet
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = OUTPUT_DIR / "enhanced_schema_example_mock.json"
        output_file.write_text(output_json, encoding="utf-8")
        print(f"\n💾 Output kaydedildi: {output_file}")
        return
    
    # LLM konfigürasyonu
    if os.getenv("GEMINI_API_KEY"):
        model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
    else:
        model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    os.environ["DRG_MODEL"] = model
    print(f"✓ LLM otomatik konfigüre edilecek: {model}\n")
    
    # Extract
    print("🔄 Extraction başlıyor...")
    try:
        entities, triples = extract_typed(text, business_schema)
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
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = OUTPUT_DIR / "enhanced_schema_example.json"
        output_file.write_text(output_json, encoding="utf-8")
        print(f"\n💾 Output kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

