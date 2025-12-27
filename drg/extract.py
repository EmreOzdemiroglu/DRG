# drg/extract.py
"""
Declarative knowledge graph extraction using DSPy.
Schema'dan dinamik olarak DSPy signature'ları oluşturur - tamamen declarative.
"""
from typing import List, Tuple, Optional, Union
import time
import logging
import json
import dspy

from .schema import (
    DRGSchema, 
    EnhancedDRGSchema, 
    Entity, 
    Relation,
    EntityType,
    RelationGroup
)

logger = logging.getLogger(__name__)


def _normalize_schema(schema: Union[DRGSchema, EnhancedDRGSchema]) -> DRGSchema:
    """Convert EnhancedDRGSchema to DRGSchema for internal use."""
    if isinstance(schema, EnhancedDRGSchema):
        return schema.to_legacy_schema()
    return schema


def _create_entity_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak EntityExtraction signature'ı oluştur."""
    normalized = _normalize_schema(schema)
    
    # Enhanced schema için daha zengin açıklama
    if isinstance(schema, EnhancedDRGSchema):
        entity_descriptions = []
        for et in schema.entity_types:
            desc = f"{et.name}: {et.description}"
            if et.examples:
                desc += f" (examples: {', '.join(et.examples[:3])})"
            entity_descriptions.append(desc)
        entity_info = "\n".join(entity_descriptions)
        entity_types = ", ".join([et.name for et in schema.entity_types])
    else:
        entity_types = ", ".join([e.name for e in normalized.entities])
        entity_info = entity_types
    
    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema."""
        text: str = dspy.InputField(desc="Input text to extract entities from")
        entities: str = dspy.OutputField(
            desc=f"JSON array of entities, each as [entity_name, entity_type]. Entity types: {entity_info}. Return only valid JSON array."
        )
    
    return EntityExtraction


def _create_relation_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak RelationExtraction signature'ı oluştur."""
    normalized = _normalize_schema(schema)
    
    # Enhanced schema için daha zengin açıklama
    if isinstance(schema, EnhancedDRGSchema):
        relation_info = []
        for rg in schema.relation_groups:
            group_desc = f"\n{rg.name}: {rg.description}"
            for r in rg.relations:
                group_desc += f"\n  - {r.name}: {r.src} -> {r.dst}"
            relation_info.append(group_desc)
        schema_info = "\n".join(relation_info)
    else:
        relation_info = []
        for r in normalized.relations:
            # Relation description varsa ekle
            if hasattr(r, 'description') and r.description:
                relation_info.append(f"{r.name} ({r.src} -> {r.dst}): {r.description}")
            else:
                relation_info.append(f"{r.name}: {r.src} -> {r.dst}")
        schema_info = "\n".join(relation_info)
    
    class RelationExtraction(dspy.Signature):
        """Extract relationships from text according to the schema."""
        text: str = dspy.InputField(desc="Input text to extract relationships from")
        entities: str = dspy.InputField(desc="JSON array of extracted entities as [[name, type], ...]")
        relations: str = dspy.OutputField(
            desc=f"JSON array of relations, each as [source, relation, target]. Allowed relations: {schema_info}. Return only valid JSON array."
        )
    
    return RelationExtraction


class KGExtractor(dspy.Module):
    """DSPy module for extracting knowledge graphs from text.
    
    Schema'dan dinamik olarak signature'lar oluşturur - tamamen declarative.
    """
    
    def __init__(self, schema: Union[DRGSchema, EnhancedDRGSchema]):
        super().__init__()
        self.schema = schema
        
        # Schema'dan dinamik signature'lar oluştur
        EntitySig = _create_entity_signature(schema)
        RelationSig = _create_relation_signature(schema)
        
        # DSPy predictor'ları oluştur
        self.entity_extractor = dspy.ChainOfThought(EntitySig)
        self.relation_extractor = dspy.ChainOfThought(RelationSig)
        
    def forward(self, text: str, max_retries: int = 3, retry_delay: float = 2.0):
        """Extract entities and relations - tamamen DSPy ile, manuel parsing yok.
        
        Args:
            text: Input text to extract from
            max_retries: Maximum number of retries on rate limit errors
            retry_delay: Delay between retries in seconds (exponential backoff)
        
        Returns:
            dspy.Prediction with entities and relations
        """
        # Step 1: Extract entities with retry logic
        logger.info("Entity extraction başlatılıyor...")
        entity_result = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Entity extraction retry {attempt + 1}/{max_retries}...")
                entity_result = self.entity_extractor(text=text)
                logger.info("Entity extraction tamamlandı")
                break
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Rate limit hit, retrying in {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        
        # Parse JSON string from DSPy output
        import json
        entities_str = entity_result.entities if hasattr(entity_result, 'entities') else "[]"
        
        # Clean markdown code blocks if present
        if isinstance(entities_str, str):
            entities_str = entities_str.strip()
            if entities_str.startswith("```json"):
                entities_str = entities_str[7:]  # Remove ```json
            elif entities_str.startswith("```"):
                entities_str = entities_str[3:]   # Remove ```
            if entities_str.endswith("```"):
                entities_str = entities_str[:-3]  # Remove trailing ```
            entities_str = entities_str.strip()
        
        try:
            entities = json.loads(entities_str) if isinstance(entities_str, str) else entities_str
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse entities JSON: {entities_str[:200]}")  # Log only first 200 chars
            entities = []
        
        # Convert to list of tuples
        entities_list = []
        for item in entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entities_list.append((str(item[0]), str(item[1])))
        
        # Step 2: Extract relations (entities'i input olarak ver) with retry logic
        logger.info(f"Relation extraction başlatılıyor ({len(entities_list)} entity ile)...")
        relation_result = None
        entities_json = json.dumps(entities_list)
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Relation extraction retry {attempt + 1}/{max_retries}...")
                relation_result = self.relation_extractor(
                    text=text,
                    entities=entities_json
                )
                logger.info("Relation extraction tamamlandı")
                break
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Rate limit hit during relation extraction, retrying in {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        
        # Parse JSON string from DSPy output
        relations_str = relation_result.relations if hasattr(relation_result, 'relations') else "[]"
        
        # Clean markdown code blocks if present
        if isinstance(relations_str, str):
            relations_str = relations_str.strip()
            if relations_str.startswith("```json"):
                relations_str = relations_str[7:]  # Remove ```json
            elif relations_str.startswith("```"):
                relations_str = relations_str[3:]   # Remove ```
            if relations_str.endswith("```"):
                relations_str = relations_str[:-3]  # Remove trailing ```
            relations_str = relations_str.strip()
        
        try:
            relations = json.loads(relations_str) if isinstance(relations_str, str) else relations_str
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse relations JSON: {relations_str[:200]}")  # Log only first 200 chars
            relations = []
        
        # Convert to list of tuples
        relations_list = []
        for item in relations:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                relations_list.append((str(item[0]), str(item[1]), str(item[2])))
        
        return dspy.Prediction(
            entities=entities_list,
            relations=relations_list
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
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Model ve API key uyumunu kontrol et
    model_lower = model.lower()
    api_key = None
    
    if "openrouter" in model_lower:
        api_key = openrouter_key
        if not api_key:
            warnings.warn(
                f"OpenRouter model ({model}) seçildi ama OPENROUTER_API_KEY bulunamadı. "
                "OpenRouter API key'i gerekli.",
                UserWarning
            )
        # OpenRouter için base URL ayarla
        if not os.getenv("DRG_BASE_URL"):
            base_url = "https://openrouter.ai/api/v1"
    elif "gemini" in model_lower:
        api_key = gemini_key
        if not api_key:
            warnings.warn(
                f"Gemini model ({model}) seçildi ama GEMINI_API_KEY bulunamadı. "
                "Gemini API key'i gerekli.",
                UserWarning
            )
    elif "anthropic" in model_lower or "claude" in model_lower:
        # OpenRouter üzerinden değilse direkt Anthropic
        api_key = anthropic_key
        if not api_key:
            warnings.warn(
                f"Anthropic model ({model}) seçildi ama ANTHROPIC_API_KEY bulunamadı. "
                "Anthropic API key'i gerekli.",
                UserWarning
            )
    elif "perplexity" in model_lower:
        api_key = perplexity_key
        if not api_key:
            warnings.warn(
                f"Perplexity model ({model}) seçildi ama PERPLEXITY_API_KEY bulunamadı. "
                "Perplexity API key'i gerekli.",
                UserWarning
            )
        # Perplexity için base URL ayarla (eğer belirtilmemişse)
        if not os.getenv("DRG_BASE_URL"):
            base_url = "https://api.perplexity.ai"
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
    # OpenRouter için özel base URL (eğer belirtilmemişse)
    if "openrouter" in model_lower and not base_url:
        base_url = "https://openrouter.ai/api/v1"
    
    # Perplexity için özel base URL (eğer belirtilmemişse)
    if "perplexity" in model_lower and not base_url:
        base_url = "https://api.perplexity.ai"
    
    # DSPy LM kwargs - temel parametreler
    lm_kwargs = {
        "model": model,
        "temperature": temperature,
    }
    
    # OpenRouter için özel konfigürasyon (LiteLLM üzerinden)
    if "openrouter" in model_lower:
        # OpenRouter için model adını doğrula
        if not model.startswith("openrouter/"):
            # Eğer prefix yoksa ekle (LiteLLM formatı)
            lm_kwargs["model"] = f"openrouter/{model}"
        else:
            # Zaten openrouter/ prefix'i var, direkt kullan
            lm_kwargs["model"] = model
        
        # LiteLLM OpenRouter için api_key ve api_base kwargs içinde geçilmeli
        if api_key:
            # Environment variable olarak set et (LiteLLM bunu otomatik okur)
            os.environ["OPENROUTER_API_KEY"] = api_key
            # Ayrıca kwargs içinde de geç (bazı durumlarda gerekebilir)
            if "kwargs" not in lm_kwargs:
                lm_kwargs["kwargs"] = {}
            lm_kwargs["kwargs"]["api_key"] = api_key
            if base_url:
                lm_kwargs["kwargs"]["api_base"] = base_url
    elif api_key:
        # Diğer servisler için api_key environment variable olarak set et
        if "gemini" in model_lower:
            os.environ["GEMINI_API_KEY"] = api_key
        elif "anthropic" in model_lower or "claude" in model_lower:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif "perplexity" in model_lower:
            os.environ["PERPLEXITY_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # kwargs içinde de geç
        if "kwargs" not in lm_kwargs:
            lm_kwargs["kwargs"] = {}
        lm_kwargs["kwargs"]["api_key"] = api_key
        if base_url:
            lm_kwargs["kwargs"]["api_base"] = base_url
    
    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    _lm_configured = True


def _get_extractor(schema: Union[DRGSchema, EnhancedDRGSchema]) -> KGExtractor:
    """Get or create extractor instance for schema."""
    global _extractor
    
    # DSPy LLM'ini otomatik konfigüre et (sadece bir kez)
    _configure_llm_auto()
    
    # Create extractor if needed or if schema changed
    if _extractor is None:
        _extractor = KGExtractor(schema)
    else:
        # Check if schema changed by comparing entity and relation sets
        normalized_old = _normalize_schema(_extractor.schema)
        normalized_new = _normalize_schema(schema)
        
        old_entities = {e.name for e in normalized_old.entities}
        old_relations = {(r.name, r.src, r.dst) for r in normalized_old.relations}
        new_entities = {e.name for e in normalized_new.entities}
        new_relations = {(r.name, r.src, r.dst) for r in normalized_new.relations}
        
        if old_entities != new_entities or old_relations != new_relations:
            _extractor = KGExtractor(schema)
    
    return _extractor


def extract_typed(text: str, schema: Union[DRGSchema, EnhancedDRGSchema]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
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
    
    # KGExtractor artık zaten list of tuples döndürüyor
    entities_typed = result.entities if hasattr(result, 'entities') and result.entities else []
    triples = result.relations if hasattr(result, 'relations') and result.relations else []
    
    # Ensure they are lists of tuples
    if not isinstance(entities_typed, list):
        entities_typed = []
    if not isinstance(triples, list):
        triples = []
    
    # Schema validation (sadece schema'ya uygun olanları döndür)
    normalized = _normalize_schema(schema)
    entity_names = {e.name for e in normalized.entities}
    valid_entities = [(name, etype) for name, etype in entities_typed if etype in entity_names]
    
    # Enhanced schema için daha gelişmiş validasyon
    if isinstance(schema, EnhancedDRGSchema):
        valid_triples = []
        for s, r, o in triples:
            s_type = next((etype for name, etype in valid_entities if name == s), None)
            o_type = next((etype for name, etype in valid_entities if name == o), None)
            
            if s_type and o_type and schema.is_valid_relation(r, s_type, o_type):
                valid_triples.append((s, r, o))
    else:
        rel_types = {(r.src, r.name, r.dst) for r in normalized.relations}
        valid_triples = []
        for s, r, o in triples:
            s_type = next((etype for name, etype in valid_entities if name == s), None)
            o_type = next((etype for name, etype in valid_entities if name == o), None)
            
            if s_type and o_type and (s_type, r, o_type) in rel_types:
                valid_triples.append((s, r, o))
    
    return valid_entities, valid_triples


# Backward-compatible thin wrapper
def extract_triples(text: str, schema: Union[DRGSchema, EnhancedDRGSchema]) -> List[Tuple[str, str, str]]:
    """Extract triples from text (backward compatibility)."""
    _, triples = extract_typed(text, schema)
    return triples


class SchemaGeneration(dspy.Signature):
    """Generate enhanced schema from text content."""
    text: str = dspy.InputField(desc="Input text to analyze for schema generation")
    generated_schema: str = dspy.OutputField(
        desc="""Generate a comprehensive EnhancedDRGSchema JSON with the following structure:

{
  "entity_types": [
    {
      "name": "EntityTypeName",
      "description": "Clear description of what this entity represents",
      "examples": ["example1", "example2", "example3"],
      "properties": {"key": "value", "key2": "value2"}
    }
  ],
  "relation_groups": [
    {
      "name": "group_name",
      "description": "What this relation group represents semantically",
      "relations": [
        {
          "name": "relation_name",
          "source": "SourceEntityType",
          "target": "TargetEntityType",
          "description": "Relationship type explanation - why this relationship exists (connection reason)",
          "detail": "Single sentence explaining why entities are connected in this specific way"
        }
      ]
    }
  ],
  "auto_discovery": true
}

**CRITICAL REQUIREMENTS FOR RICH KNOWLEDGE GRAPHS:**

1. **Entity Types**:
   - Extract ALL relevant entity types from the text (not just main entities)
   - Include supporting entities: locations, technologies, processes, organizations, etc.
   - For each entity type, provide 3-5 real examples from the text
   - Add meaningful properties that characterize each entity type

2. **Relations - MOST IMPORTANT**:
   - **DO NOT create a star-shaped graph** (all relations from one central entity)
   - **Create a rich, interconnected graph** with relations between ALL entity types
   - For EACH pair of entity types, consider what relationships might exist between them
   - Examples of entity-to-entity relations:
     * Vehicle → Vehicle: "Succeeded_By", "Shares_Platform_With", "Competes_With", "Similar_To"
     * Technology → Technology: "Supports", "Integrates_With", "Enhances", "Depends_On", "Works_With"
     * Vehicle → Technology: "Equipped_With", "Has_Feature", "Receives", "Uses"
     * Technology → Vehicle: "Enables", "Installed_In", "Used_By"
     * Manufacturing → Technology: "Produces", "Uses", "Develops"
     * Technology → Manufacturing: "Used_In", "Enables"
     * Manufacturing → Location: "Located_In", "Operates_In"
     * Vehicle → Manufacturing: "Produced_At", "Manufactured_In"
     * Manufacturing → Vehicle: "Produces", "Manufactures"
     * Entity → Industry: "Impacts", "Transforms", "Disrupts"
     * Industry → Industry: "Part_Of", "Related_To", "Competes_With"

3. **Relation Groups**:
   - Group related relations semantically (e.g., 'production', 'technology', 'location', 'temporal', 'hierarchical', 'social', 'professional', 'spatial', 'conceptual')
   - Each group should contain multiple relations covering different entity type pairs
   - Aim for 5-8 relation groups with 4-10 relations each
   - Total target: 30-50+ relations across all groups

4. **Relation Details**:
   - For each relation, provide:
     * description: The reason/type of connection (what kind of relationship)
     * detail: Specific detail about why/how entities are connected in this way

5. **Graph Structure - CRITICAL**:
   - Target: 60-70% of relations should be BETWEEN entities (not from central entity)
   - Include diverse relation patterns: hierarchical, sequential, dependency, spatial, temporal, functional, social, professional
   - For EVERY pair of entity types, consider what relationships might exist between them
   - Create a rich, interconnected web - NOT a star-shaped graph
   - Ensure the schema supports extracting a comprehensive, meaningful knowledge graph

6. **Comprehensive Relation Coverage**:
   - **Person ↔ Person**: Collaborates_With, Works_With, Assists, Mentors, Reports_To, Related_To, Opposes
   - **Person ↔ Organization**: Works_At, Leads, Member_Of, Founded, Owns, Opposes, Monitored_By
   - **Organization ↔ Organization**: Competes_With, Collaborates_With, Acquires, Merges_With, Opposes, Partners_With
   - **Technology ↔ Technology**: Based_On, Evolved_From, Integrates_With, Supports, Replaces, Contains, Enables, Works_With
   - **Entity ↔ Location**: Located_In, Works_In, Operates_In, Relocated_To, Houses, Near, Beneath, Above
   - **Entity ↔ Concept**: Represents, Promotes, Enhances, Threatens, Protects, Ensures, Fights_Against, Related_To, Contradicts
   - **Technology ↔ Person**: Developed_By, Used_By, Benefits, Controlled_By, Monitored_By
   - **Technology ↔ Organization**: Developed_At, Owned_By, Deployed_At, Seeks_To_Control

7. **Schema Completeness**:
   - Aim for 30-50+ relations total across 4-8 relation groups
   - Each entity type should participate in multiple relation types (both as source and target)
   - Include bidirectional thinking: if A → B exists, consider if B → A also makes sense

**Example Rich Schema Structure:**
If text mentions a company with products, technologies, and locations:
- Entity Types: Company, Product, Technology, Location, Manufacturing, Person, Industry
- Relations (40+):
  * Company → Product (Introduced, Produces, Markets)
  * Company → Technology (Develops, Owns, Uses, Deploys)
  * Company → Location (Located_In, Operates_In, Expands_To)
  * Product → Product (Succeeded_By, Similar_To, Competes_With, Shares_Platform_With)
  * Product → Technology (Equipped_With, Uses, Requires, Powered_By)
  * Technology → Technology (Supports, Integrates_With, Based_On, Enables, Replaces)
  * Manufacturing → Location (Located_In, Operates_In)
  * Manufacturing → Product (Produces, Manufactures)
  * Manufacturing → Technology (Uses, Develops, Produces)
  * Product → Location (Produced_In, Sold_In, Available_In)
  * Person → Company (Works_At, Founded, Leads, Owns)
  * Person → Product (Designed_By, Developed_By)
  * Person → Technology (Invented_By, Uses, Benefits)
  * Company → Industry (Part_Of, Dominates, Transforms)
  ... and many more to create a rich graph

Return ONLY valid JSON, no extra text or markdown formatting."""
    )


def generate_schema_from_text(text: str, max_retries: int = 3, retry_delay: float = 2.0) -> EnhancedDRGSchema:
    """
    Metinden otomatik olarak enhanced şema oluştur.
    
    Bu fonksiyon, verilen metni analiz ederek uygun entity tipleri (properties ve examples ile),
    relation grupları ve detaylı açıklamalar çıkarır ve bir EnhancedDRGSchema nesnesi döndürür.
    
    Args:
        text: Analiz edilecek metin
        max_retries: Rate limit hatası durumunda maksimum deneme sayısı
        retry_delay: Denemeler arası bekleme süresi (saniye)
    
    Returns:
        EnhancedDRGSchema: Metne uygun detaylı şema
    """
    # LLM'i konfigüre et
    _configure_llm_auto()
    
    # Metni örnekleme için kısalt (çok uzunsa - şema oluşturmak için yeterli context için)
    # İlk, orta ve son kısmı alarak daha kapsamlı bir örnek oluştur (entity'ler arası ilişkileri yakalamak için)
    if len(text) > 15000:
        # Çok uzun metinler için: başlangıç, ilk orta, ikinci orta, son (4 parça)
        part_size = len(text) // 4
        sample_text = (
            text[:3500] + "\n\n[... truncated ...]\n\n" +
            text[part_size:part_size+3500] + "\n\n[... truncated ...]\n\n" +
            text[part_size*2:part_size*2+3500] + "\n\n[... truncated ...]\n\n" +
            text[-3500:]
        )
        logger.info(f"Metin çok uzun ({len(text)} karakter), dört parça kullanılıyor (başlangıç, orta-1, orta-2, son)...")
    elif len(text) > 12000:
        # Üç parça al: başlangıç, orta, son (entity'ler arası ilişkiler genelde orta kısımda olur)
        part_size = len(text) // 3
        sample_text = text[:4000] + "\n\n[... truncated ...]\n\n" + text[part_size:part_size+4000] + "\n\n[... truncated ...]\n\n" + text[-4000:]
        logger.info(f"Metin uzun ({len(text)} karakter), üç parça kullanılıyor (başlangıç, orta, son)...")
    elif len(text) > 8000:
        sample_text = text[:4000] + "\n\n[... truncated ...]\n\n" + text[-4000:]
        logger.info(f"Metin uzun ({len(text)} karakter), ilk ve son 4000 karakteri kullanılıyor...")
    else:
        sample_text = text
    
    # Schema generation signature'ı oluştur
    schema_generator = dspy.ChainOfThought(SchemaGeneration)
    
    # Şema oluştur (retry logic ile)
    schema_result = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Schema generation retry {attempt + 1}/{max_retries}...")
            schema_result = schema_generator(text=sample_text)
            logger.info("Schema generation tamamlandı")
            break
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit during schema generation, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit error after {max_retries} attempts")
                    raise
            else:
                raise
    
    # Parse JSON schema
    schema_str = schema_result.generated_schema if hasattr(schema_result, 'generated_schema') else "{}"
    
    # Clean markdown code blocks if present
    if isinstance(schema_str, str):
        # Remove markdown code block markers
        schema_str = schema_str.strip()
        if schema_str.startswith("```json"):
            schema_str = schema_str[7:]  # Remove ```json
        elif schema_str.startswith("```"):
            schema_str = schema_str[3:]   # Remove ```
        if schema_str.endswith("```"):
            schema_str = schema_str[:-3]  # Remove trailing ```
        schema_str = schema_str.strip()
    
    try:
        schema_data = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse schema JSON: {schema_str}")
        logger.error(f"JSON parse error: {e}")
        # Fallback: varsayılan enhanced şema
        logger.warning("Varsayılan enhanced şema kullanılıyor...")
        return _create_default_enhanced_schema()
    
    # Enhanced schema formatını parse et
    entity_types = []
    for et_data in schema_data.get("entity_types", []):
        if isinstance(et_data, dict) and "name" in et_data:
            entity_types.append(EntityType(
                name=et_data["name"],
                description=et_data.get("description", ""),
                examples=et_data.get("examples", []),
                properties=et_data.get("properties", {})
            ))
    
    # Relation groups oluştur
    relation_groups = []
    for rg_data in schema_data.get("relation_groups", []):
        if isinstance(rg_data, dict) and "name" in rg_data:
            relations = []
            for r_data in rg_data.get("relations", []):
                if isinstance(r_data, dict) and all(k in r_data for k in ["name", "source", "target"]):
                    relations.append(Relation(
                        name=r_data["name"],
                        src=r_data["source"],
                        dst=r_data["target"],
                        description=r_data.get("description", ""),
                        detail=r_data.get("detail", "")
                    ))
            
            if relations:
                relation_groups.append(RelationGroup(
                    name=rg_data["name"],
                    description=rg_data.get("description", ""),
                    relations=relations,
                    examples=rg_data.get("examples", [])
                ))
    
    # Legacy format desteği (geriye dönük uyumluluk)
    if not entity_types and "entities" in schema_data:
        for e_data in schema_data.get("entities", []):
            if isinstance(e_data, dict) and "name" in e_data:
                entity_types.append(EntityType(
                    name=e_data["name"],
                    description=e_data.get("description", ""),
                    examples=e_data.get("examples", []),
                    properties=e_data.get("properties", {})
                ))
    
    if not relation_groups and "relations" in schema_data:
        # Legacy relations'ı tek bir group'a ekle
        relations = []
        for r_data in schema_data.get("relations", []):
            if isinstance(r_data, dict) and all(k in r_data for k in ["name", "source", "target"]):
                relations.append(Relation(
                    name=r_data["name"],
                    src=r_data["source"],
                    dst=r_data["target"],
                    description=r_data.get("description", ""),
                    detail=r_data.get("detail", "")
                ))
        
        if relations:
            relation_groups.append(RelationGroup(
                name="general",
                description="General relations",
                relations=relations
            ))
    
    # En az bir entity type ve relation group olmalı
    if not entity_types:
        logger.warning("Şemada entity type bulunamadı, varsayılan entity type'lar ekleniyor...")
        entity_types = [
            EntityType(
                name="Person",
                description="Individuals, people mentioned in the text",
                examples=[],
                properties={}
            ),
            EntityType(
                name="Location",
                description="Geographic locations, places",
                examples=[],
                properties={}
            ),
            EntityType(
                name="Event",
                description="Events, occurrences",
                examples=[],
                properties={}
            )
        ]
    
    if not relation_groups:
        logger.warning("Şemada relation group bulunamadı, varsayılan relation'lar ekleniyor...")
        entity_names = {et.name for et in entity_types}
        relations = []
        if "Person" in entity_names and len(entity_names) >= 2:
            if "Location" in entity_names:
                relations.append(Relation(
                    name="located_in",
                    src="Person",
                    dst="Location",
                    description="Person is located in a location",
                    detail="Geographic or physical location relationship"
                ))
            if "Event" in entity_names:
                relations.append(Relation(
                    name="participated_in",
                    src="Person",
                    dst="Event",
                    description="Person participates in an event",
                    detail="Participation or involvement in an event"
                ))
        
        if relations:
            relation_groups.append(RelationGroup(
                name="general",
                description="General relations",
                relations=relations
            ))
    
    try:
        schema = EnhancedDRGSchema(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=True
        )
        
        # Schema kalitesi kontrolü - entity'ler arası ilişki oranını kontrol et
        entity_names = {et.name for et in entity_types}
        all_relations = []
        first_entity = list(entity_names)[0] if entity_names else None
        central_entity_relations = 0
        cross_entity_relations = 0
        
        for rg in relation_groups:
            for r in rg.relations:
                all_relations.append(r)
                # İlk entity type'ı merkezi entity olarak kabul et (genelde Company/Organization)
                if r.src == first_entity:
                    central_entity_relations += 1
                else:
                    cross_entity_relations += 1
        
        total_relations = len(all_relations)
        total_relations_count = sum(len(rg.relations) for rg in relation_groups)
        
        if total_relations > 0:
            cross_ratio = cross_entity_relations / total_relations
            if cross_ratio < 0.4:
                logger.warning(
                    f"⚠️  Schema kalite uyarısı: Entity'ler arası ilişki oranı düşük ({cross_ratio:.1%}). "
                    f"Schema çoğunlukla merkezi entity'den çıkan ilişkiler içeriyor. "
                    f"KG zenginliği için daha fazla cross-entity relation eklenmeli. "
                    f"Önerilen: %60-70 cross-entity relations."
                )
            elif cross_ratio >= 0.6:
                logger.info(f"✅ Schema kalitesi iyi: {cross_ratio:.1%} cross-entity relations ({cross_entity_relations}/{total_relations})")
            else:
                logger.info(f"ℹ️  Schema kalitesi orta: {cross_ratio:.1%} cross-entity relations ({cross_entity_relations}/{total_relations})")
        
        logger.info(f"Enhanced schema oluşturuldu: {len(entity_types)} entity type, {len(relation_groups)} relation group, {total_relations_count} relation")
        return schema
    except ValueError as e:
        logger.error(f"Schema validation hatası: {e}")
        # Hatalı relation'ları filtrele ve tekrar dene
        entity_names = {et.name for et in entity_types}
        valid_relation_groups = []
        for rg in relation_groups:
            valid_relations = [
                r for r in rg.relations 
                if r.src in entity_names and r.dst in entity_names
            ]
            if valid_relations:
                valid_relation_groups.append(RelationGroup(
                    name=rg.name,
                    description=rg.description,
                    relations=valid_relations,
                    examples=rg.examples
                ))
        
        if valid_relation_groups:
            return EnhancedDRGSchema(
                entity_types=entity_types,
                relation_groups=valid_relation_groups,
                auto_discovery=True
            )
        else:
            # Son çare: varsayılan enhanced şema
            logger.warning("Hatalı şema, varsayılan enhanced şema kullanılıyor...")
            return _create_default_enhanced_schema()


def _create_default_enhanced_schema() -> EnhancedDRGSchema:
    """Varsayılan enhanced şema oluştur."""
    entity_types = [
        EntityType(
            name="Person",
            description="Individuals, people mentioned in the text",
            examples=[],
            properties={}
        ),
        EntityType(
            name="Location",
            description="Geographic locations, places",
            examples=[],
            properties={}
        ),
        EntityType(
            name="Event",
            description="Events, occurrences",
            examples=[],
            properties={}
        )
    ]
    
    relation_groups = [
        RelationGroup(
            name="general",
            description="General relations",
            relations=[
                Relation(
                    name="related_to",
                    src="Person",
                    dst="Person",
                    description="Person is related to another person",
                    detail="Family, social, or professional relationship"
                ),
                Relation(
                    name="located_in",
                    src="Person",
                    dst="Location",
                    description="Person is located in a location",
                    detail="Geographic or physical location relationship"
                ),
                Relation(
                    name="participated_in",
                    src="Person",
                    dst="Event",
                    description="Person participates in an event",
                    detail="Participation or involvement in an event"
                )
            ]
        )
    ]
    
    return EnhancedDRGSchema(
        entity_types=entity_types,
        relation_groups=relation_groups,
        auto_discovery=True
    )
