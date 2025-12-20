#!/usr/bin/env python3
"""
Uzun metin testi - Anayasa Mahkemesi kararı.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from drg import (
    EntityType,
    Relation,
    RelationGroup,
    EnhancedDRGSchema,
    extract_typed,
    KG,
)

# Anayasa Mahkemesi Kararı
LONG_TEXT = """
TÜRKİYE CUMHURİYETİ ANAYASA MAHKEMESİ BİRİNCİ BÖLÜM KARAR

İÇİM NURAY YANMAZ BAŞVURUSU (Başvuru Numarası: 2023/15469)
Karar Tarihi: 5/11/2025

Başkan: Hasan Tahsin GÖKCAN
Üyeler: Recai AKYEL, Selahaddin MENTEŞ, Muhterem İNCE, Yılmaz AKÇİL
Raportör: Volkan SEVTEKİN
Başvurucu: İçim Nuray YANMAZ

I. BAŞVURUNUN ÖZETİ

Başvuru 8/3/2012 tarihli ve 6284 sayılı Ailenin Korunması ve Kadına Karşı Şiddetin 
Önlenmesine Dair Kanun uyarınca verilen tedbir kararına yönelik esaslı iddiaların 
itiraz mercii tarafından karşılanmaması nedeniyle gerekçeli karar hakkının ihlal 
edildiği iddiasına ilişkindir.

6284 sayılı Kanun uyarınca başvurucu aleyhine tedbir talep edilmiştir. Mahkeme, 
tedbir talebinin kabulüne karar vermiştir. Başvurucunun karara karşı yaptığı itiraz 
kesin olarak reddedilmiştir.

Başvurucu, nihai hükmü 2/2/2023 tarihinde öğrendikten sonra 22/2/2023 tarihinde 
süresi içerisinde bireysel başvuruda bulunmuştur.

II. DEĞERLENDİRME

Başvurucu 6284 sayılı Kanun uyarınca verilen tedbir kararına yönelik esaslı 
iddialarının itiraz mercii tarafından karşılanmaması nedeniyle gerekçeli karar 
hakkının ihlal edildiğini ileri sürmüştür.

Anayasa Mahkemesi, gerekçeli karar hakkı yönünden olay ve olguları somut başvuru 
ile benzer iddiaları Salih Söylemezoğlu (B. No: 2013/3758, 6/1/2016) ve 
Erdal Türkmen (B. No: 2016/2100, 4/4/2019) ve S.M. (B. No: 2016/6038, 20/6/2019) 
kararlarında incelemiştir.

Başvuruya konu olayda lehine tedbir isteyenlerin başvurucunun annesi ve kardeşleri 
oldukları ve şiddete uğrama tehlikesi altında bulunduklarını iddia ettikleri 
görülmektedir. Bu kapsamda mahkemece tedbir isteyenlerin ısrarlı takip mağduru 
oldukları kabul edilerek 6284 sayılı Kanun'un bazı hükümlerinin tedbiren 
uygulanması gerektiği kanaatine varılmıştır.

Açıklanan gerekçelerle başvurucunun Anayasa'nın 36. maddesinde güvence altına 
alınan gerekçeli karar hakkının ihlal edildiğine karar verilmesi gerekir.

IV. HÜKÜM

A. Gerekçeli karar hakkının ihlal edildiğine ilişkin iddianın KABUL EDİLEBİLİR OLDUĞUNA,
B. Anayasa'nın 36. maddesinde güvence altına alınan adil yargılanma hakkı kapsamındaki 
   gerekçeli karar hakkının İHLAL EDİLDİĞİNE,
C. Kararın bir örneğinin yeniden yargılama yapılması için Alanya 2. Aile Mahkemesine 
   (E.2023/13 D.İş, K.2023/12) iletilmek üzere Alanya 1. Aile Mahkemesine 
   (E.2023/59 D.İş, K.2023/60) GÖNDERİLMESİNE,
D. Başvurucunun tazminat talebinin REDDİNE,
E. 1.480,40 TL harçtan oluşan yargılama giderinin başvurucuya ÖDENMESİNE,
F. Kararın bir örneğinin Adalet Bakanlığına GÖNDERİLMESİNE 5/11/2025 tarihinde 
   OYBİRLİĞİYLE karar verildi.
"""


def build_schema() -> EnhancedDRGSchema:
    """Anayasa Mahkemesi kararları için şema."""
    return EnhancedDRGSchema(
        entity_types=[
            EntityType(
                name="Court",
                description="Mahkeme",
                examples=["Anayasa Mahkemesi", "Alanya 1. Aile Mahkemesi"],
            ),
            EntityType(
                name="Person",
                description="Kişi (hakim, başvurucu, raportör)",
                examples=["İçim Nuray YANMAZ", "Hasan Tahsin GÖKCAN"],
            ),
            EntityType(
                name="Law",
                description="Kanun veya Anayasa maddesi",
                examples=["6284 sayılı Kanun", "Anayasa m.36"],
            ),
            EntityType(
                name="CourtDecision",
                description="Mahkeme kararı veya emsal",
                examples=["2023/15469", "B. No: 2013/3758"],
            ),
            EntityType(
                name="Right",
                description="Hukuki hak",
                examples=["gerekçeli karar hakkı", "adil yargılanma hakkı"],
            ),
            EntityType(
                name="Organization",
                description="Kurum",
                examples=["Adalet Bakanlığı"],
            ),
        ],
        relation_groups=[
            RelationGroup(
                name="court_relations",
                description="Mahkeme ilişkileri",
                relations=[
                    Relation("decided_by", "CourtDecision", "Court"),
                    Relation("role_in", "Person", "Court"),
                    Relation("applicant_of", "Person", "CourtDecision"),
                ],
            ),
            RelationGroup(
                name="legal_citations",
                description="Hukuki atıflar",
                relations=[
                    Relation("cites_law", "CourtDecision", "Law"),
                    Relation("cites_decision", "CourtDecision", "CourtDecision"),
                    Relation("protects_right", "Law", "Right"),
                ],
            ),
            RelationGroup(
                name="procedural",
                description="Usul ilişkileri",
                relations=[
                    Relation("sent_to", "CourtDecision", "Court"),
                    Relation("sent_to_org", "CourtDecision", "Organization"),
                ],
            ),
        ],
        auto_discovery=True,
    )


def has_api_key() -> bool:
    return bool(
        os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )


def main():
    print("=" * 70)
    print("📜 Uzun Metin Testi - Anayasa Mahkemesi Kararı")
    print("=" * 70)
    
    print(f"\n📊 Metin uzunluğu: {len(LONG_TEXT)} karakter, ~{len(LONG_TEXT.split())} kelime\n")
    
    schema = build_schema()
    summary = schema.get_schema_summary()
    print(f"Schema: {len(summary['entity_types'])} entity type, {len(summary['relation_groups'])} relation group\n")
    
    if not has_api_key():
        print("⚠️  API key yok - test atlanıyor")
        print("   OPENAI_API_KEY veya GEMINI_API_KEY set edin")
        return
    
    # Model config
    if os.getenv("GEMINI_API_KEY"):
        model = os.getenv("DRG_MODEL", "gemini/gemini-2.0-flash-exp")
    else:
        model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    os.environ["DRG_MODEL"] = model
    print(f"🤖 Model: {model}\n")
    
    print("🔄 Extraction başlıyor...")
    try:
        entities, triples = extract_typed(LONG_TEXT, schema)
        triples = list(dict.fromkeys(triples))
        
        print(f"\n✅ {len(entities)} entity, {len(triples)} relation bulundu\n")
        
        print("📌 Entities:")
        for name, etype in entities:
            print(f"   [{etype}] {name}")
        
        print("\n🔗 Relations:")
        for s, r, o in triples:
            print(f"   {s} --[{r}]--> {o}")
        
        kg = KG.from_typed(entities, triples)
        
        # Save output
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "long_text_test.json"
        output_file.write_text(kg.to_json(), encoding="utf-8")
        print(f"\n💾 Saved: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()




