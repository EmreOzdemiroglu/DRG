# drg/extract.py
"""
Declarative knowledge graph extraction using DSPy.
Schema'dan dinamik olarak DSPy signature'ları oluşturur - tamamen declarative.
"""
from typing import List, Tuple, Optional
import dspy

from .schema import DRGSchema


def _create_entity_signature(schema: DRGSchema) -> type:
    """Schema'dan dinamik olarak EntityExtraction signature'ı oluştur."""
    entity_types = ", ".join([e.name for e in schema.entities])
    
    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema."""
        text: str = dspy.InputField(desc="Input text to extract entities from")
        entities: List[Tuple[str, str]] = dspy.OutputField(
            desc=f"List of (entity_name, entity_type) tuples. Entity types: {entity_types}"
        )
    
    return EntityExtraction


def _create_relation_signature(schema: DRGSchema) -> type:
    """Schema'dan dinamik olarak RelationExtraction signature'ı oluştur."""
    relation_info = []
    for r in schema.relations:
        relation_info.append(f"{r.name}: {r.src} -> {r.dst}")
    schema_info = "; ".join(relation_info)
    
    class RelationExtraction(dspy.Signature):
        """Extract relationships from text according to the schema."""
        text: str = dspy.InputField(desc="Input text to extract relationships from")
        entities: List[Tuple[str, str]] = dspy.InputField(desc="List of extracted entities (name, type)")
        relations: List[Tuple[str, str, str]] = dspy.OutputField(
            desc=f"List of (source, relation, target) triples. Allowed relations: {schema_info}"
        )
    
    return RelationExtraction


class KGExtractor(dspy.Module):
    """DSPy module for extracting knowledge graphs from text.
    
    Schema'dan dinamik olarak signature'lar oluşturur - tamamen declarative.
    """
    
    def __init__(self, schema: DRGSchema):
        super().__init__()
        self.schema = schema
        
        # Schema'dan dinamik signature'lar oluştur
        EntitySig = _create_entity_signature(schema)
        RelationSig = _create_relation_signature(schema)
        
        # DSPy predictor'ları oluştur
        self.entity_extractor = dspy.ChainOfThought(EntitySig)
        self.relation_extractor = dspy.ChainOfThought(RelationSig)
        
    def forward(self, text: str):
        """Extract entities and relations - tamamen DSPy ile, manuel parsing yok."""
        # Step 1: Extract entities
        entity_result = self.entity_extractor(text=text)
        
        # DSPy otomatik olarak structured output döndürür
        entities = entity_result.entities if hasattr(entity_result, 'entities') else []
        
        # Step 2: Extract relations (entities'i input olarak ver)
        relation_result = self.relation_extractor(
            text=text,
            entities=entities
        )
        
        # DSPy otomatik olarak structured output döndürür
        relations = relation_result.relations if hasattr(relation_result, 'relations') else []
        
        return dspy.Prediction(
            entities=entities,
            relations=relations
        )


# Global extractor instance (lazy initialized)
_extractor: Optional[KGExtractor] = None
_lm_configured = False


def _configure_llm_auto():
    """DSPy LLM'ini otomatik olarak environment variable'lardan konfigüre et."""
    global _lm_configured
    
    if _lm_configured:
        return
    
    import os
    import warnings
    
    # Environment variable'lardan otomatik oku
    model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    
    # API key'leri environment'tan oku
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Model ve API key uyumunu kontrol et
    model_lower = model.lower()
    api_key = None
    
    if "gemini" in model_lower:
        api_key = gemini_key
        if not api_key:
            warnings.warn(
                f"Gemini model ({model}) seçildi ama GEMINI_API_KEY bulunamadı. "
                "Gemini API key'i gerekli.",
                UserWarning
            )
    elif "anthropic" in model_lower or "claude" in model_lower:
        api_key = anthropic_key
        if not api_key:
            warnings.warn(
                f"Anthropic model ({model}) seçildi ama ANTHROPIC_API_KEY bulunamadı. "
                "Anthropic API key'i gerekli.",
                UserWarning
            )
    elif "ollama" in model_lower:
        # Ollama için API key gerekmez
        api_key = None
    else:
        # OpenAI veya diğer modeller için
        api_key = openai_key
        if not api_key and not model_lower.startswith("ollama"):
            warnings.warn(
                f"Cloud model ({model}) seçildi ama OPENAI_API_KEY bulunamadı. "
                "API key gerekli olabilir.",
                UserWarning
            )
    
    base_url = os.getenv("DRG_BASE_URL")
    temperature = float(os.getenv("DRG_TEMPERATURE", "0.0"))
    
    # DSPy LM'ini konfigüre et
    lm_kwargs = {
        "model": model,
        "temperature": temperature,
    }
    
    if api_key:
        lm_kwargs["api_key"] = api_key
    
    if base_url:
        lm_kwargs["api_base"] = base_url
    
    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    _lm_configured = True


def _get_extractor(schema: DRGSchema) -> KGExtractor:
    """Get or create extractor instance for schema."""
    global _extractor
    
    # DSPy LLM'ini otomatik konfigüre et (sadece bir kez)
    _configure_llm_auto()
    
    # Create extractor if needed or if schema changed
    if _extractor is None:
        _extractor = KGExtractor(schema)
    else:
        # Check if schema changed by comparing entity and relation sets
        old_entities = {e.name for e in _extractor.schema.entities}
        old_relations = {(r.name, r.src, r.dst) for r in _extractor.schema.relations}
        new_entities = {e.name for e in schema.entities}
        new_relations = {(r.name, r.src, r.dst) for r in schema.relations}
        
        if old_entities != new_entities or old_relations != new_relations:
            _extractor = KGExtractor(schema)
    
    return _extractor


def extract_typed(text: str, schema: DRGSchema) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations from text using DSPy.
    
    Tamamen declarative - sadece schema tanımlıyorsun, DSPy gerisini hallediyor.
    
    Args:
        text: Input text to extract from
        schema: DRGSchema defining allowed entity types and relations
    
    Returns:
        Tuple of (entities_typed, triples) where:
        - entities_typed: List of (entity_name, entity_type) tuples
        - triples: List of (source, relation, target) tuples
    """
    extractor = _get_extractor(schema)
    result = extractor(text=text)
    
    # DSPy structured output'dan direkt al (manuel parsing yok)
    entities = result.entities if hasattr(result, 'entities') and result.entities else []
    relations = result.relations if hasattr(result, 'relations') and result.relations else []
    
    # Convert to tuples if needed (DSPy bazen list döndürebilir)
    entities_typed = []
    for item in entities:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            entities_typed.append((str(item[0]), str(item[1])))
    
    triples = []
    for item in relations:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triples.append((str(item[0]), str(item[1]), str(item[2])))
    
    # Schema validation (sadece schema'ya uygun olanları döndür)
    entity_names = {e.name for e in schema.entities}
    valid_entities = [(name, etype) for name, etype in entities_typed if etype in entity_names]
    
    rel_types = {(r.src, r.name, r.dst) for r in schema.relations}
    valid_triples = []
    for s, r, o in triples:
        s_type = next((etype for name, etype in valid_entities if name == s), None)
        o_type = next((etype for name, etype in valid_entities if name == o), None)
        
        if s_type and o_type and (s_type, r, o_type) in rel_types:
            valid_triples.append((s, r, o))
    
    return valid_entities, valid_triples


# Backward-compatible thin wrapper
def extract_triples(text: str, schema: DRGSchema) -> List[Tuple[str, str, str]]:
    """Extract triples from text (backward compatibility)."""
    _, triples = extract_typed(text, schema)
    return triples
