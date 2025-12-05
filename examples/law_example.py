#!/usr/bin/env python3
"""
Türk hukuku odaklı DRG örneği.
Koşum: `uv run python examples/law_example.py`
"""

import os
import sys
from pathlib import Path

# Proje root'unu path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import (
    EntityType,
    Relation,
    RelationGroup,
    EnhancedDRGSchema,
    extract_typed,
    KG,
)


OUTPUT_DIR = Path("outputs")


def has_api_key() -> bool:
    """Env'de bir LLM anahtarı var mı?"""
    return bool(
        os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )


def build_schema() -> EnhancedDRGSchema:
    """Türk hukukuna yönelik örnek şema."""
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Law",
                description="Kanun veya mevzuat başlığı",
                examples=["6698 sayılı KVKK", "5237 sayılı TCK", "6098 sayılı TBK"],
            ),
            EntityType(
                name="Article",
                description="Belirli bir kanun maddesi",
                examples=["KVKK m.12", "KVKK m.7", "TCK 142"],
            ),
            EntityType(
                name="Court",
                description="Mahkeme veya yüksek mahkeme",
                examples=["Anayasa Mahkemesi", "Yargıtay", "Danıştay"],
            ),
            EntityType(
                name="CourtDecision",
                description="Belirli bir karar veya dosya",
                examples=["2024/115 E., 2025/12 K.", "2023/45 D. sayılı karar"],
            ),
            EntityType(
                name="Organization",
                description="Şirket veya kurum",
                examples=["ACME Teknoloji A.Ş.", "BTK", "Banka A.Ş."],
            ),
            EntityType(
                name="Right",
                description="Kanunun tanıdığı hak",
                examples=["kişisel verilerin silinmesi hakkı", "veri güvenliği hakkı"],
            ),
            EntityType(
                name="Obligation",
                description="Kanundan doğan yükümlülük",
                examples=["veri güvenliği yükümlülüğü", "aydınlatma yükümlülüğü"],
            ),
            EntityType(
                name="Sanction",
                description="İdari para cezası veya yaptırım",
                examples=["idari para cezası", "erişim engeli"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="citations",
                description="Karar ve kanun/madde atıfları",
                relations=[
                    Relation("cites_law", "CourtDecision", "Law"),
                    Relation("cites_article", "CourtDecision", "Article"),
                    Relation("article_of", "Article", "Law"),
                ],
                examples=[
                    {
                        "text": "AYM, KVKK m.12'ye atıf yaptı.",
                        "entities": [
                            ("AYM kararı", "CourtDecision"),
                            ("KVKK m.12", "Article"),
                            ("6698 sayılı KVKK", "Law"),
                        ],
                        "relations": [
                            ("AYM kararı", "cites_article", "KVKK m.12"),
                            ("KVKK m.12", "article_of", "6698 sayılı KVKK"),
                        ],
                    }
                ],
            ),
            RelationGroup(
                name="adjudication",
                description="Karar, mahkeme ve taraf etkisi",
                relations=[
                    Relation("decided_by", "CourtDecision", "Court"),
                    Relation("affects", "CourtDecision", "Organization"),
                    Relation("imposes_sanction", "CourtDecision", "Sanction"),
                ],
            ),
            RelationGroup(
                name="rights_obligations",
                description="Hak ve yükümlülük ilişkileri",
                relations=[
                    Relation("grants_right", "Law", "Right"),
                    Relation("implements_right", "Article", "Right"),
                    Relation("imposes_obligation", "Article", "Obligation"),
                    Relation("obligation_on", "Obligation", "Organization"),
                ],
            ),
        ],
        auto_discovery=True,
    )


def main():
    print("=" * 70)
    print("⚖️  Türk Hukuku DRG Örneği")
    print("=" * 70)
    print("Koşum: uv run python examples/law_example.py\n")
    
    schema = build_schema()
    summary = schema.get_schema_summary()
    print(f"Schema: {len(summary['entity_types'])} entity type, {len(summary['relation_groups'])} relation group")
    
    # Test metni
    text = (
        "Anayasa Mahkemesi 2024/115 E., 2025/12 K. sayılı kararında 6698 sayılı Kişisel "
        "Verilerin Korunması Kanunu'nun 12. maddesi kapsamında veri güvenliği yükümlülüğünü "
        "ihlal eden ACME Teknoloji A.Ş. hakkında verilen idari para cezasını onadı. "
        "Kararda KVKK m.7'deki kişisel verilerin silinmesi hakkına da atıf yapıldı."
    )
    print("\n📄 Test Metni:")
    print(f"   {text}\n")
    
    if not has_api_key():
        print("⚠️  API key yok - mock verisi ile KG gösteriliyor\n")
        entities = [
            ("6698 sayılı KVKK", "Law"),
            ("KVKK m.12", "Article"),
            ("KVKK m.7", "Article"),
            ("Anayasa Mahkemesi", "Court"),
            ("2024/115 E., 2025/12 K.", "CourtDecision"),
            ("ACME Teknoloji A.Ş.", "Organization"),
            ("veri güvenliği yükümlülüğü", "Obligation"),
            ("kişisel verilerin silinmesi hakkı", "Right"),
            ("idari para cezası", "Sanction"),
        ]
        triples = [
            ("KVKK m.12", "article_of", "6698 sayılı KVKK"),
            ("KVKK m.7", "article_of", "6698 sayılı KVKK"),
            ("2024/115 E., 2025/12 K.", "decided_by", "Anayasa Mahkemesi"),
            ("2024/115 E., 2025/12 K.", "cites_article", "KVKK m.12"),
            ("2024/115 E., 2025/12 K.", "cites_article", "KVKK m.7"),
            ("2024/115 E., 2025/12 K.", "affects", "ACME Teknoloji A.Ş."),
            ("2024/115 E., 2025/12 K.", "imposes_sanction", "idari para cezası"),
            ("KVKK m.12", "imposes_obligation", "veri güvenliği yükümlülüğü"),
            ("veri güvenliği yükümlülüğü", "obligation_on", "ACME Teknoloji A.Ş."),
            ("KVKK m.7", "implements_right", "kişisel verilerin silinmesi hakkı"),
            ("6698 sayılı KVKK", "grants_right", "kişisel verilerin silinmesi hakkı"),
        ]
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = OUTPUT_DIR / "law_example_mock.json"
        output_file.write_text(output_json, encoding="utf-8")
        print("📊 Knowledge Graph (Mock):")
        print(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
        return
    
    # LLM konfigürasyonu (OpenAI varsayılan, Gemini varsa onu kullan)
    if os.getenv("GEMINI_API_KEY"):
        model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
    else:
        model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    os.environ["DRG_MODEL"] = model
    print(f"✓ LLM otomatik konfigüre edilecek: {model}\n")
    
    print("🔄 Extraction başlıyor...")
    try:
        entities, triples = extract_typed(text, schema)
        triples = list(dict.fromkeys(triples))  # remove dups while preserving order
        
        print(f"\n✅ {len(entities)} entity, {len(triples)} relation bulundu")
        for name, etype in entities:
            print(f"  - {name} ({etype})")
        print()
        for s, r, o in triples:
            print(f"  - {s} --[{r}]--> {o}")
        
        kg = KG.from_typed(entities, triples)
        output_json = kg.to_json()
        OUTPUT_DIR.mkdir(exist_ok=True)
        output_file = OUTPUT_DIR / "law_example.json"
        output_file.write_text(output_json, encoding="utf-8")
        
        print("\n📊 Knowledge Graph JSON:")
        print(output_json)
        print(f"\n💾 Output kaydedildi: {output_file}")
    except Exception as exc:
        print(f"\n❌ Hata: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

