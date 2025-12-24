#!/usr/bin/env python3
"""
GraphRAG Pipeline Example - Tam Pipeline Testi

GraphRAG yapısına uygun olarak:
1. Chunking
2. KG Extraction
3. Entity Embedding'leri Ekleme
4. Community Report Oluşturma
5. GraphRAG Retrieval (KG traversal + Community reports)

Kullanım:
    python examples/graphrag_pipeline_example.py [example_name]

Örnek:
    python examples/graphrag_pipeline_example.py 1
    python examples/graphrag_pipeline_example.py 1example
    python examples/graphrag_pipeline_example.py example1  # Otomatik 1example'a çevrilir
"""

import os
import sys
import json
import ssl
from pathlib import Path

# SSL sorununu çöz (sandbox ortamında /Library/Frameworks/ erişim kısıtlaması için)
# Environment variables
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# SSL context'i unverified olarak ayarla
ssl._create_default_https_context = ssl._create_unverified_context

# requests adapters SSL context yükleme işlemini bypass et
try:
    import requests
    from requests import adapters
    if hasattr(adapters, '_preloaded_ssl_context'):
        try:
            unverified_context = ssl.create_default_context()
            unverified_context.check_hostname = False
            unverified_context.verify_mode = ssl.CERT_NONE
            adapters._preloaded_ssl_context = unverified_context
        except:
            adapters._preloaded_ssl_context = None
except:
    pass

# urllib3 SSL uyarılarını kapat
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

# Proje root'u path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Union

from drg.chunking import create_chunker
from drg.embedding import create_embedding_provider
from drg.extract import extract_typed, KGExtractor
from drg.schema import DRGSchema, EnhancedDRGSchema, Entity, Relation, EntityType, RelationGroup
from drg.graph import KG
from drg.graph.kg_core import EnhancedKG, KGNode, KGEdge
from drg.graph.community_report import CommunityReportGenerator
from drg.clustering import create_clustering_algorithm
from drg.retrieval import create_graphrag_retriever


def load_schema(schema_path: str) -> Union[DRGSchema, EnhancedDRGSchema]:
    """Schema'yı JSON dosyasından yükle (Enhanced veya legacy format destekler)."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_data = json.load(f)
    
    # Enhanced schema formatını kontrol et
    if "entity_types" in schema_data:
        from drg.schema import EntityType, RelationGroup
        entity_types = [
            EntityType(
                name=et["name"],
                description=et.get("description", ""),
                examples=et.get("examples", []),
                properties=et.get("properties", {})
            )
            for et in schema_data["entity_types"]
        ]
        
        relation_groups = []
        for rg_data in schema_data.get("relation_groups", []):
            relations = [
                Relation(
                    name=r["name"],
                    src=r["source"],
                    dst=r["target"],
                    description=r.get("description", ""),
                    detail=r.get("detail", "")
                )
                for r in rg_data.get("relations", [])
            ]
            relation_groups.append(RelationGroup(
                name=rg_data["name"],
                description=rg_data.get("description", ""),
                relations=relations,
                examples=rg_data.get("examples", [])
            ))
        
        return EnhancedDRGSchema(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=schema_data.get("auto_discovery", False)
        )
    else:
        # Legacy format
        entities = [Entity(e["name"]) for e in schema_data.get("entities", [])]
        relations = [
            Relation(
                r["name"], 
                r.get("source", r.get("src", "")), 
                r.get("target", r.get("dst", "")),
                description=r.get("description", ""),
                detail=r.get("detail", "")
            )
            for r in schema_data.get("relations", [])
        ]
        
        return DRGSchema(entities=entities, relations=relations)


def save_schema(schema: Union[DRGSchema, EnhancedDRGSchema], schema_path: str) -> None:
    """Schema'yı JSON dosyasına kaydet."""
    from drg.schema import EnhancedDRGSchema as EDRGS
    from pathlib import Path
    
    schema_path_obj = Path(schema_path)
    schema_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(schema, EDRGS):
        # Enhanced schema formatı
        schema_dict = {
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "examples": et.examples,
                    "properties": et.properties
                }
                for et in schema.entity_types
            ],
            "relation_groups": [
                {
                    "name": rg.name,
                    "description": rg.description,
                    "relations": [
                        {
                            "name": r.name,
                            "source": r.src,
                            "target": r.dst,
                            "description": r.description,
                            "detail": r.detail
                        }
                        for r in rg.relations
                    ],
                    "examples": rg.examples
                }
                for rg in schema.relation_groups
            ],
            "auto_discovery": schema.auto_discovery
        }
    else:
        # Legacy format
        schema_dict = {
            "entities": [{"name": e.name} for e in schema.entities],
            "relations": [
                {
                    "name": r.name,
                    "source": r.src,
                    "target": r.dst,
                    "description": r.description,
                    "detail": r.detail
                }
                for r in schema.relations
            ]
        }
    
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=2, ensure_ascii=False)


def main():
    """GraphRAG pipeline'ını test et."""
    
    # Example name belirle - sayı başta formatı kullan (1example, 2example, etc.)
    raw_name = sys.argv[1] if len(sys.argv) > 1 else "1"
    
    # Eğer sadece sayı verilmişse "Nexample" formatına çevir
    if raw_name.isdigit():
        example_name = f"{raw_name}example"
    elif raw_name.startswith("example"):
        # Eski format: "example1" -> "1example"
        num = raw_name.replace("example", "")
        example_name = f"{num}example" if num.isdigit() else raw_name
    else:
        example_name = raw_name
    
    print("=" * 70)
    print(f"GraphRAG Pipeline Test - {example_name.upper()}")
    print("=" * 70)
    
    # Klasör yapısını oluştur
    inputs_dir = Path("inputs")
    outputs_dir = Path("outputs")
    inputs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # Schema ve text dosya yolları - sayı başta format
    schema_path = inputs_dir / f"{example_name}_schema.json"
    text_path = inputs_dir / f"{example_name}_text.txt"
    
    # API Key ayarla (sadece environment variable'dan oku)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        print("✅ OPENROUTER_API_KEY environment variable'dan okundu")
    else:
        print("⚠️  OPENROUTER_API_KEY bulunamadı. OpenRouter servisleri çalışmayabilir.")
        print("   API key'i ayarlamak için: export OPENROUTER_API_KEY='your-key'")
    if not os.getenv("DRG_MODEL"):
        os.environ["DRG_MODEL"] = "openrouter/anthropic/claude-3-haiku"
    
    # Metni dosyadan yükle
    if not text_path.exists():
        print(f"⚠️  Metin dosyası bulunamadı: {text_path}")
        print(f"   Lütfen {text_path} dosyasını oluşturun")
        return
    
    print(f"📄 Metin dosyası yükleniyor: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"\n📄 Metin uzunluğu: {len(text)} karakter")
    print(f"   Kelime sayısı: {len(text.split())}")
    
    # Step 1: Chunking
    print("\n" + "=" * 70)
    print("1. CHUNKING")
    print("=" * 70)
    
    # Declarative chunking: preset kullanarak
    chunker = create_chunker(preset="graphrag")
    
    chunks = chunker.chunk(
        text=text,
        origin_dataset="apple_corpus",
        origin_file="apple_history.txt",
    )
    
    print(f"✅ {len(chunks)} chunk oluşturuldu")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"   Chunk {i}: {chunk.token_count} tokens, ID: {chunk.chunk_id}")
    
    # Step 2: KG Extraction
    print("\n" + "=" * 70)
    print("2. KNOWLEDGE GRAPH EXTRACTION")
    print("=" * 70)
    
    # Schema'yı dosyadan yükle veya metinden otomatik oluştur
    if not schema_path.exists():
        print(f"⚠️  Schema dosyası bulunamadı: {schema_path}")
        print(f"   Metinden otomatik schema oluşturuluyor...")
        # LLM'i yapılandır (schema generation öncesi)
        from drg.extract import _configure_llm_auto, generate_schema_from_text
        _configure_llm_auto()
        print("   🔧 LLM konfigürasyonu yapıldı")
        print("   🤖 LLM ile metin analiz ediliyor ve uygun şema oluşturuluyor...")
        
        # Metinden otomatik şema oluştur (EnhancedDRGSchema döndürür)
        schema = generate_schema_from_text(text)
        
        # Oluşturulan şemayı kaydet
        save_schema(schema, str(schema_path))
        print(f"   ✅ Otomatik enhanced schema oluşturuldu ve kaydedildi: {schema_path}")
        if isinstance(schema, EnhancedDRGSchema):
            total_relations = sum(len(rg.relations) for rg in schema.relation_groups)
            print(f"      {len(schema.entity_types)} entity type, {len(schema.relation_groups)} relation group, {total_relations} relation")
        else:
            print(f"      {len(schema.entities)} entity, {len(schema.relations)} relation")
    else:
        print(f"📄 Schema yükleniyor: {schema_path}")
        schema = load_schema(schema_path)
        if isinstance(schema, EnhancedDRGSchema):
            total_relations = sum(len(rg.relations) for rg in schema.relation_groups)
            print(f"   ✅ Enhanced schema yüklendi: {len(schema.entity_types)} entity type, {len(schema.relation_groups)} relation group, {total_relations} relation")
        else:
            print(f"   ✅ Legacy schema yüklendi: {len(schema.entities)} entity, {len(schema.relations)} relation")
        # LLM'i yapılandır (extraction öncesi)
        print("   🔧 LLM konfigürasyonu yapılıyor...")
        from drg.extract import _configure_llm_auto
        _configure_llm_auto()
        print("   ✅ LLM konfigürasyonu tamamlandı")
    
    try:
        print("   🔄 Chunk-based KG extraction başlatılıyor...")
        print(f"   📊 {len(chunks)} chunk üzerinde extraction yapılacak")
        extractor = KGExtractor(schema)
        
        # Her chunk için extraction yap ve birleştir
        all_entities = set()  # Duplicate'leri önlemek için set kullan
        all_triples = set()
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   🔍 Chunk {i}/{len(chunks)} işleniyor ({chunk.token_count} tokens)...")
            try:
                result = extractor.forward(chunk.text)
                
                # Parse entities and relations
                entities_str = result.entities if hasattr(result, 'entities') else "[]"
                relations_str = result.relations if hasattr(result, 'relations') else "[]"
                
                entities = json.loads(entities_str) if isinstance(entities_str, str) else entities_str
                relations = json.loads(relations_str) if isinstance(relations_str, str) else relations_str
                
                # Convert to tuples ve ekle
                for e in entities:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        all_entities.add((str(e[0]), str(e[1])))
                
                for r in relations:
                    if isinstance(r, (list, tuple)) and len(r) >= 3:
                        all_triples.add((str(r[0]), str(r[1]), str(r[2])))
                
                print(f"      ✅ {len([e for e in entities if isinstance(e, (list, tuple)) and len(e) >= 2])} entity, "
                      f"{len([r for r in relations if isinstance(r, (list, tuple)) and len(r) >= 3])} relation bulundu")
            except Exception as e:
                print(f"      ⚠️  Chunk {i} extraction hatası: {e}")
                continue
        
        # Set'leri list'e çevir
        entities_list = list(all_entities)
        triples_list = list(all_triples)
        
        print(f"✅ Toplam {len(entities_list)} unique entity ve {len(triples_list)} unique relation extract edildi")
        print(f"   Örnek entities: {entities_list[:5]}")
        print(f"   Örnek relations: {triples_list[:3]}")
        
    except Exception as e:
        print(f"⚠️  KG extraction hatası: {e}")
        print("   Mock data kullanılıyor...")
        entities_list = [
            ("Apple", "Company"),
            ("Steve Jobs", "Person"),
            ("Tim Cook", "Person"),
            ("iPhone", "Product"),
            ("iPad", "Product"),
            ("Mac", "Product"),
            ("Cupertino", "Location"),
            ("2007", "Year"),
        ]
        triples_list = [
            ("Apple", "founded_by", "Steve Jobs"),
            ("Apple", "ceo_of", "Tim Cook"),
            ("Apple", "produces", "iPhone"),
            ("Apple", "produces", "iPad"),
            ("Apple", "produces", "Mac"),
            ("Apple", "located_in", "Cupertino"),
            ("iPhone", "released_in", "2007"),
        ]
        print(f"   Mock: {len(entities_list)} entity ve {len(triples_list)} relation")
    
    # Step 3: EnhancedKG Oluştur
    print("\n" + "=" * 70)
    print("3. ENHANCED KG OLUŞTURMA")
    print("=" * 70)
    
    enhanced_kg = EnhancedKG()
    
    # Nodes ekle
    for entity_id, entity_type in entities_list:
        node = KGNode(
            id=entity_id,
            type=entity_type,
            properties={},
            metadata={},
        )
        enhanced_kg.add_node(node)
    
    # Relation description ve detail'lerini schema'dan al
    relation_descriptions = {}
    relation_details = {}
    
    if isinstance(schema, EnhancedDRGSchema):
        # Enhanced schema'dan relation description ve detail'lerini al
        for rg in schema.relation_groups:
            for r in rg.relations:
                if r.description:
                    relation_descriptions[r.name] = r.description
                if r.detail:
                    relation_details[r.name] = r.detail
    else:
        # Legacy schema'dan
        for r in schema.relations:
            if hasattr(r, 'description') and r.description:
                relation_descriptions[r.name] = r.description
            if hasattr(r, 'detail') and r.detail:
                relation_details[r.name] = r.detail
    
    # Edges ekle
    for source, relation, target in triples_list:
        # Self-loop kontrolü: source ve target aynıysa atla
        if source == target:
            continue
        
        # Source ve target node'ları yoksa ekle
        if source not in enhanced_kg.nodes:
            enhanced_kg.add_node(KGNode(id=source, type=None))
        if target not in enhanced_kg.nodes:
            enhanced_kg.add_node(KGNode(id=target, type=None))
        
        # Relationship detail: Schema'daki detail varsa kullan, yoksa description, yoksa basit format
        if relation in relation_details:
            relationship_detail = relation_details[relation]
        elif relation in relation_descriptions:
            relationship_detail = relation_descriptions[relation]
        else:
            relationship_detail = f"{source} {relation} {target}"
        
        try:
            edge = KGEdge(
                source=source,
                target=target,
                relationship_type=relation,
                relationship_detail=relationship_detail,
                metadata={},
            )
            enhanced_kg.add_edge(edge)
        except ValueError as e:
            # Self-loop veya diğer validation hatalarını atla
            print(f"   ⚠️  Edge atlandı ({source} --[{relation}]--> {target}): {e}")
            continue
    
    print(f"✅ EnhancedKG oluşturuldu: {len(enhanced_kg.nodes)} node, {len(enhanced_kg.edges)} edge")
    
    # Step 4: Entity Embedding'leri Ekle
    print("\n" + "=" * 70)
    print("4. ENTITY EMBEDDING'LERİ EKLEME")
    print("=" * 70)
    
    # Embedding provider oluştur
    try:
        # OpenRouter üzerinden embedding (eğer destekleniyorsa)
        embedding_provider = create_embedding_provider(
            provider="openrouter",
            model="text-embedding-3-small",  # OpenAI embedding via OpenRouter
        )
        print("   OpenRouter embedding provider kullanılıyor")
    except Exception as e:
        print(f"   OpenRouter embedding yok ({e}), local embedding deneniyor...")
        try:
            embedding_provider = create_embedding_provider(
                provider="local",
                model="sentence-transformers/all-MiniLM-L6-v2",
            )
            print("   Local embedding provider kullanılıyor")
        except Exception as e2:
            print(f"   Local embedding yok ({e2}), mock embedding kullanılıyor...")
            class MockEmbeddingProvider:
                def __init__(self):
                    import hashlib
                    self.hash_func = hashlib.md5
                
                def embed(self, text: str):
                    """Deterministic embedding based on text hash."""
                    import random
                    seed = int(self.hash_func(text.encode()).hexdigest()[:8], 16)
                    random.seed(seed)
                    return [random.random() for _ in range(384)]
                
                def embed_batch(self, texts):
                    return [self.embed(text) for text in texts]
                
                def get_dimension(self):
                    return 384
                
                def get_model_name(self):
                    return "mock/deterministic"
            
            embedding_provider = MockEmbeddingProvider()
            print("   Mock embedding provider kullanılıyor (deterministic)")
    
    # Entity embedding'lerini ekle
    entity_texts = {node_id: node_id for node_id in enhanced_kg.nodes.keys()}
    enhanced_kg.add_entity_embeddings(embedding_provider, entity_texts)
    
    embedded_count = sum(1 for node in enhanced_kg.nodes.values() if node.embedding is not None)
    print(f"✅ {embedded_count}/{len(enhanced_kg.nodes)} node'a embedding eklendi")
    
    # Step 5: Clustering ve Community Reports
    print("\n" + "=" * 70)
    print("5. CLUSTERING VE COMMUNITY REPORTS")
    print("=" * 70)
    
    try:
        # Clustering yap
        clustering_algorithm = create_clustering_algorithm(
            algorithm="louvain",
        )
        
        # KG'yi NetworkX formatına çevir (clustering için)
        import networkx as nx
        G = nx.Graph()
        for node_id in enhanced_kg.nodes.keys():
            G.add_node(node_id)
        for edge in enhanced_kg.edges:
            G.add_edge(edge.source, edge.target)
        
        # Clustering yap
        clustering_result = clustering_algorithm.cluster(G)
        
        # EnhancedKG'ye cluster'ları ekle
        # clustering_result is List[Cluster] from drg.clustering.algorithms
        from drg.graph.kg_core import Cluster as KGCluster
        for clustering_cluster in clustering_result:
            # Convert clustering Cluster to KG Cluster
            kg_cluster = KGCluster(
                id=f"cluster_{clustering_cluster.cluster_id}",
                node_ids=set(clustering_cluster.nodes),
                metadata=clustering_cluster.metadata,
            )
            enhanced_kg.add_cluster(kg_cluster)
        
        print(f"✅ {len(clustering_result)} cluster oluşturuldu")
        
        # Community reports oluştur
        report_generator = CommunityReportGenerator(enhanced_kg)
        reports = report_generator.generate_all_reports()
        
        print(f"✅ {len(reports)} community report oluşturuldu")
        for report in reports[:2]:
            print(f"   - {report.cluster_id}: {len(report.top_actors)} actors, {len(report.top_relationships)} relations")
        
    except Exception as e:
        print(f"⚠️  Clustering hatası: {e}")
        print("   Community reports olmadan devam ediliyor...")
        reports = []
    
    # Step 6: GraphRAG Retrieval
    print("\n" + "=" * 70)
    print("6. GRAPHRAG RETRIEVAL")
    print("=" * 70)
    
    try:
        retriever = create_graphrag_retriever(
            embedding_provider=embedding_provider,
            knowledge_graph=enhanced_kg,
            community_reports=reports if reports else None,
            similarity_threshold=0.3,  # Düşük threshold (mock embeddings için)
            max_hops=2,
        )
        
        # Test query'leri
        queries = [
            "What products does Apple produce?",
            "Who is the CEO of Apple?",
            "Where is Apple located?",
        ]
        
        for query in queries:
            print(f"\n   Query: {query}")
            context = retriever.retrieve(query, k_entities=5, k_reports=3)
            
            print(f"   ✅ Seed entities: {context.seed_entities}")
            print(f"   ✅ Subgraph: {len(context.kg_subgraph['nodes'])} nodes, {len(context.kg_subgraph['edges'])} edges")
            print(f"   ✅ Entities: {len(context.entities)}")
            print(f"   ✅ Relationships: {len(context.relationships)}")
            print(f"   ✅ Community reports: {len(context.community_reports)}")
            
            if context.entities:
                print(f"      Entities: {', '.join(context.entities[:5])}")
            if context.relationships:
                print(f"      Sample relation: {context.relationships[0]['source']} --[{context.relationships[0]['relationship_type']}]--> {context.relationships[0]['target']}")
    
    except Exception as e:
        print(f"⚠️  GraphRAG retrieval hatası: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Output'ları Kaydet
    print("\n" + "=" * 70)
    print("7. OUTPUT'LARI KAYDETME")
    print("=" * 70)
    
    # Output dosya yolları
    kg_output_path = outputs_dir / f"{example_name}_kg.json"
    summary_output_path = outputs_dir / f"{example_name}_summary.json"
    reports_output_path = outputs_dir / f"{example_name}_community_reports.json"
    schema_output_path = outputs_dir / f"{example_name}_schema.json"
    
    # Schema'yı output olarak kaydet (kullanılan schema - Enhanced veya Legacy)
    save_schema(schema, str(schema_output_path))
    print(f"✅ Schema kaydedildi: {schema_output_path}")
    
    # KG'yi kaydet
    kg_json = enhanced_kg.to_json(indent=2)
    with open(kg_output_path, "w", encoding="utf-8") as f:
        f.write(kg_json)
    print(f"✅ KG kaydedildi: {kg_output_path}")
    
    # Community reports'u kaydet
    if reports:
        report_generator.export_reports_json(reports, str(reports_output_path))
        print(f"✅ Community reports kaydedildi: {reports_output_path}")
    else:
        print(f"ℹ️  Community reports yok (clustering gerekli)")
    
    # Pipeline summary
    summary = {
        "example_name": example_name,
        "input_files": {
            "schema": str(schema_path),
            "text": str(text_path),
        },
        "chunks": len(chunks),
        "entities": len(entities_list),
        "relations": len(triples_list),
        "kg_nodes": len(enhanced_kg.nodes),
        "kg_edges": len(enhanced_kg.edges),
        "embedded_nodes": embedded_count,
        "clusters": len(enhanced_kg.clusters),
        "community_reports": len(reports),
        "output_files": {
            "schema": str(schema_output_path),
            "kg": str(kg_output_path),
            "summary": str(summary_output_path),
            "community_reports": str(reports_output_path) if reports else None,
        },
    }
    
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Pipeline summary kaydedildi: {summary_output_path}")
    
    print("\n" + "=" * 70)
    print("✅ GRAPHRAG PIPELINE TAMAMLANDI!")
    print("=" * 70)
    print(f"\n📊 Özet:")
    print(f"   - Chunks: {summary['chunks']}")
    print(f"   - KG Nodes: {summary['kg_nodes']}")
    print(f"   - KG Edges: {summary['kg_edges']}")
    print(f"   - Embedded Nodes: {summary['embedded_nodes']}")
    print(f"   - Clusters: {summary['clusters']}")
    print(f"   - Community Reports: {summary['community_reports']}")


if __name__ == "__main__":
    main()

