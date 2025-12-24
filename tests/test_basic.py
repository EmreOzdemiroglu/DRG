"""
DRG Test Suite - Tüm provider'lar için entity ve relation extraction testleri.
"""
import os
import pytest
from drg.schema import Entity, Relation, DRGSchema
from drg.extract import extract_triples
from drg.graph.kg_core import EnhancedKG, KGNode, KGEdge


def _get_test_schema() -> DRGSchema:
    """Test için standart schema oluştur."""
    return DRGSchema(
        entities=[
            Entity("Company"),
            Entity("Product"),
            Entity("Person")
        ],
        relations=[
            Relation("produces", "Company", "Product", description="Şirket ürün üretir"),
            Relation("founded_by", "Company", "Person", description="Şirket kişi tarafından kuruldu"),
            Relation("ceo_of", "Person", "Company", description="Kişi şirketin CEO'sudur")
        ]
    )


def _get_test_text() -> str:
    """Test için standart metin."""
    return "Apple Inc. was founded by Steve Jobs in 1976. Tim Cook is the current CEO of Apple. Apple produces the iPhone, iPad, and Mac computers."


def _check_api_key_and_set_model(provider: str) -> str:
    """
    Provider'a göre API key kontrolü yap ve model ayarla.
    
    Args:
        provider: "openai", "gemini", "openrouter", "anthropic"
    
    Returns:
        Model adı
    
    Raises:
        pytest.skip: API key yoksa test'i atla
    """
    model_map = {
        "openai": "openai/gpt-4o-mini",
        "gemini": "gemini/gemini-2.0-flash-exp",
        "openrouter": "openrouter/anthropic/claude-3-haiku",
        "anthropic": "anthropic/claude-3-haiku"
    }
    
    api_key_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }
    
    model = model_map.get(provider)
    api_key_env = api_key_map.get(provider)
    
    if not model or not api_key_env:
        pytest.skip(f"Unknown provider: {provider}")
    
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(
            f"No {api_key_env} found. "
            f"Set {api_key_env} environment variable to run this test."
        )
    
    # Model'i environment variable'a set et
    os.environ["DRG_MODEL"] = model
    
    return model


def test_extract_entities_and_relations_with_openai():
    """OpenAI model ile entity ve relation extraction testi."""
    model = _check_api_key_and_set_model("openai")
    
    schema = _get_test_schema()
    text = _get_test_text()
    
    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))  # Duplicate'leri kaldır
    
    # EnhancedKG oluştur
    enhanced_kg = EnhancedKG()
    
    # Entities ekle
    entity_map = {}
    for source, relation, target in triples:
        if source not in entity_map:
            entity_map[source] = KGNode(id=source, type=None)
            enhanced_kg.add_node(entity_map[source])
        if target not in entity_map:
            entity_map[target] = KGNode(id=target, type=None)
            enhanced_kg.add_node(entity_map[target])
    
    # Edges ekle
    for source, relation, target in triples:
        edge = KGEdge(
            source=source,
            target=target,
            relationship_type=relation,
            relationship_detail=f"{source} {relation} {target}",
            metadata={}
        )
        enhanced_kg.add_edge(edge)
    
    # Assertions
    assert len(enhanced_kg.nodes) > 0, "En az bir node olmalı"
    assert len(enhanced_kg.edges) > 0, "En az bir edge olmalı"
    
    # Apple entity'si olmalı
    assert "Apple" in enhanced_kg.nodes, "Apple entity'si bulunmalı"
    
    # iPhone veya iPad gibi bir product olmalı
    product_found = any(
        "iPhone" in node_id or "iPad" in node_id or "Mac" in node_id
        for node_id in enhanced_kg.nodes.keys()
    )
    assert product_found, "En az bir product entity'si bulunmalı"


def test_extract_entities_and_relations_with_gemini():
    """Gemini model ile entity ve relation extraction testi."""
    model = _check_api_key_and_set_model("gemini")
    
    schema = _get_test_schema()
    text = _get_test_text()
    
    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))
    
    # EnhancedKG oluştur
    enhanced_kg = EnhancedKG()
    
    # Entities ve edges ekle
    entity_map = {}
    for source, relation, target in triples:
        if source not in entity_map:
            entity_map[source] = KGNode(id=source, type=None)
            enhanced_kg.add_node(entity_map[source])
        if target not in entity_map:
            entity_map[target] = KGNode(id=target, type=None)
            enhanced_kg.add_node(entity_map[target])
        
        edge = KGEdge(
            source=source,
            target=target,
            relationship_type=relation,
            relationship_detail=f"{source} {relation} {target}",
            metadata={}
        )
        enhanced_kg.add_edge(edge)
    
    # Assertions
    assert len(enhanced_kg.nodes) > 0
    assert len(enhanced_kg.edges) > 0
    assert "Apple" in enhanced_kg.nodes


def test_extract_entities_and_relations_with_openrouter():
    """OpenRouter model ile entity ve relation extraction testi."""
    model = _check_api_key_and_set_model("openrouter")
    
    schema = _get_test_schema()
    text = _get_test_text()
    
    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))
    
    # EnhancedKG oluştur
    enhanced_kg = EnhancedKG()
    
    # Entities ve edges ekle
    entity_map = {}
    for source, relation, target in triples:
        if source not in entity_map:
            entity_map[source] = KGNode(id=source, type=None)
            enhanced_kg.add_node(entity_map[source])
        if target not in entity_map:
            entity_map[target] = KGNode(id=target, type=None)
            enhanced_kg.add_node(entity_map[target])
        
        edge = KGEdge(
            source=source,
            target=target,
            relationship_type=relation,
            relationship_detail=f"{source} {relation} {target}",
            metadata={}
        )
        enhanced_kg.add_edge(edge)
    
    # Assertions
    assert len(enhanced_kg.nodes) > 0
    assert len(enhanced_kg.edges) > 0
    assert "Apple" in enhanced_kg.nodes


def test_extract_entities_and_relations_with_anthropic():
    """Anthropic (Claude) model ile entity ve relation extraction testi."""
    model = _check_api_key_and_set_model("anthropic")
    
    schema = _get_test_schema()
    text = _get_test_text()
    
    triples = extract_triples(text, schema)
    triples = list(dict.fromkeys(triples))
    
    # EnhancedKG oluştur
    enhanced_kg = EnhancedKG()
    
    # Entities ve edges ekle
    entity_map = {}
    for source, relation, target in triples:
        if source not in entity_map:
            entity_map[source] = KGNode(id=source, type=None)
            enhanced_kg.add_node(entity_map[source])
        if target not in entity_map:
            entity_map[target] = KGNode(id=target, type=None)
            enhanced_kg.add_node(entity_map[target])
        
        edge = KGEdge(
            source=source,
            target=target,
            relationship_type=relation,
            relationship_detail=f"{source} {relation} {target}",
            metadata={}
        )
        enhanced_kg.add_edge(edge)
    
    # Assertions
    assert len(enhanced_kg.nodes) > 0
    assert len(enhanced_kg.edges) > 0
    assert "Apple" in enhanced_kg.nodes


def test_schema_with_relation_descriptions():
    """Schema'daki relation description'larının doğru yüklendiğini test et."""
    schema = _get_test_schema()
    
    # Relation description'ları kontrol et
    produces_rel = next((r for r in schema.relations if r.name == "produces"), None)
    assert produces_rel is not None, "produces relation bulunmalı"
    assert hasattr(produces_rel, 'description'), "Relation'da description field'ı olmalı"
    assert produces_rel.description == "Şirket ürün üretir", "Description doğru yüklenmeli"
    
    founded_by_rel = next((r for r in schema.relations if r.name == "founded_by"), None)
    assert founded_by_rel is not None
    assert founded_by_rel.description == "Şirket kişi tarafından kuruldu"


def test_enhanced_kg_structure():
    """EnhancedKG yapısının doğru çalıştığını test et."""
    enhanced_kg = EnhancedKG()
    
    # Node ekle
    node1 = KGNode(id="Apple", type="Company")
    node2 = KGNode(id="iPhone", type="Product")
    enhanced_kg.add_node(node1)
    enhanced_kg.add_node(node2)
    
    # Edge ekle
    edge = KGEdge(
        source="Apple",
        target="iPhone",
        relationship_type="produces",
        relationship_detail="Apple iPhone üretir",
        metadata={}
    )
    enhanced_kg.add_edge(edge)
    
    # Assertions
    assert len(enhanced_kg.nodes) == 2
    assert len(enhanced_kg.edges) == 1
    assert "Apple" in enhanced_kg.nodes
    assert "iPhone" in enhanced_kg.nodes
    
    # Edge kontrolü
    assert edge.source == "Apple"
    assert edge.target == "iPhone"
    assert edge.relationship_type == "produces"
    assert "iPhone" in edge.relationship_detail
