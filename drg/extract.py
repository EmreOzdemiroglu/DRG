# drg/extract.py
"""
Declarative knowledge graph extraction using DSPy.
Schema'dan dinamik olarak DSPy signature'ları oluşturur - tamamen declarative.
"""
from typing import List, Tuple, Optional, Union, Any, Dict
import logging
import json
import dspy
from pydantic import BaseModel

from .utils.llm_throttle import throttle_llm_calls

from .schema import (
    DRGSchema, 
    EnhancedDRGSchema, 
    Entity, 
    Relation,
    EntityType,
    RelationGroup
)

logger = logging.getLogger(__name__)


# Pydantic models for structured output (DSPy TypedPredictor compatibility)
class EntityList(BaseModel):
    """Structured output model for entity extraction."""
    entities: List[Tuple[str, str]]  # List of (entity_name, entity_type) tuples


class RelationList(BaseModel):
    """Structured output model for relation extraction."""
    relations: List[Tuple[str, str, str]]  # List of (source, relation, target) tuples


def _parse_json_output(json_str: str, expected_format: str = "array") -> list:
    """Parse JSON string from DSPy output.
    
    NOTE: This function is DEPRECATED and used only in fallback mode when TypedPredictor 
    is not available. In normal operation, TypedPredictor handles JSON parsing automatically
    without requiring manual string manipulation.
    
    For new code, use TypedPredictor with Pydantic models instead of this function.
    This function exists only for backward compatibility with older DSPy versions.
    
    Args:
        json_str: JSON string to parse (may include markdown code blocks in legacy mode)
        expected_format: Expected JSON format ("array" or "object")
    
    Returns:
        Parsed JSON data (list or dict, depending on expected_format)
    
    Raises:
        ValueError: If JSON parsing fails (instead of silently returning empty result)
    """
    if not isinstance(json_str, str):
        error_msg = f"Expected string, got {type(json_str).__name__}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # NOTE: Manual markdown code block cleaning is a workaround for legacy DSPy usage
    # with dspy.Predict (not TypedPredictor). TypedPredictor handles this automatically.
    # This is kept only for backward compatibility.
    # Try parsing directly first (most LLMs output clean JSON when using TypedPredictor-style prompts)
    json_str = json_str.strip()
    
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # If direct parsing fails, try cleaning markdown code blocks (legacy LLM behavior)
        # This is a known issue with some LLMs wrapping JSON in markdown code blocks
        # when not using TypedPredictor's structured output handling
        cleaned = json_str
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            logger.debug("JSON parsing succeeded after markdown code block removal (legacy behavior)")
        except json.JSONDecodeError as e:
<<<<<<< HEAD
            # As a last resort, accept Python-literal dict/list strings (common LLM failure mode),
            # but parse them safely via ast.literal_eval (no code execution).
            try:
                import ast
                parsed = ast.literal_eval(cleaned)
                logger.debug("Parsed output via ast.literal_eval fallback (python-literal JSON)")
            except Exception:
                error_msg = (
                    f"Failed to parse JSON output even after markdown cleaning: {str(e)}. "
                    f"Input: {json_str[:200]}"
                )
=======
            error_msg = f"Failed to parse JSON output even after markdown cleaning: {str(e)}. Input: {json_str[:200]}"
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    # Validate format - raise error if format doesn't match expected (don't silently return empty)
    if expected_format == "array" and not isinstance(parsed, list):
        error_msg = f"Expected JSON array, got {type(parsed).__name__}. Input: {json_str[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif expected_format == "object" and not isinstance(parsed, dict):
        error_msg = f"Expected JSON object, got {type(parsed).__name__}. Input: {json_str[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return parsed


def _normalize_schema(schema: Union[DRGSchema, EnhancedDRGSchema]) -> DRGSchema:
    """Convert EnhancedDRGSchema to DRGSchema for internal use."""
    if isinstance(schema, EnhancedDRGSchema):
        return schema.to_legacy_schema()
    return schema


def _create_entity_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak EntityExtraction signature'ı oluştur.
    
    DSPy best practice: Minimal signature with clear field descriptions.
    Structured output will be handled by TypedPredictor with Pydantic model.
    """
    normalized = _normalize_schema(schema)
    
    # Enhanced schema için daha zengin açıklama
    if isinstance(schema, EnhancedDRGSchema):
        entity_types = ", ".join([et.name for et in schema.entity_types])
        entity_descriptions = []
        for et in schema.entity_types:
            desc = f"{et.name}: {et.description}"
            if et.examples:
                desc += f" (examples: {', '.join(et.examples[:3])})"
            entity_descriptions.append(desc)
        entity_info = "\n".join(entity_descriptions)
    else:
        entity_types = ", ".join([e.name for e in normalized.entities])
        entity_info = entity_types
    
    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema.
        
        Extract all entities of the specified types from the input text.
        Entity types: {entity_types}
        """
        text: str = dspy.InputField(desc="Input text to extract entities from")
        # Output will be handled by TypedPredictor with EntityList Pydantic model
        # No OutputField here - TypedPredictor handles structured output
    
    # Replace docstring with formatted version (DSPy uses __doc__ attribute for prompt generation)
    # This ensures actual entity types are included in the prompt, not placeholders
    EntityExtraction.__doc__ = f"""Extract entities from text according to the schema.

Extract all entities of the specified types from the input text.
Entity types: {entity_types}
"""
    
    # Store entity info in class for later use (optional, for debugging/validation)
    EntityExtraction._entity_info = entity_info
    EntityExtraction._entity_types = entity_types
    
    return EntityExtraction


def _create_relation_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak RelationExtraction signature'ı oluştur.
    
    DSPy best practice: Minimal signature with clear field descriptions.
    Structured output will be handled by TypedPredictor with Pydantic model.
    """
    normalized = _normalize_schema(schema)
    
    # Enhanced schema için daha zengin açıklama
    if isinstance(schema, EnhancedDRGSchema):
        relation_info = []
        for rg in schema.relation_groups:
            group_desc = f"{rg.name}: {rg.description}"
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
        """Extract relationships from text according to the schema.
        
        Extract all relationships between the provided entities based on the schema.
        Allowed relations: {schema_info}
        """
        text: str = dspy.InputField(desc="Input text to extract relationships from")
        entities: List[Tuple[str, str]] = dspy.InputField(
            desc="List of extracted entities as [(name, type), ...] tuples"
        )
        # Output will be handled by TypedPredictor with RelationList Pydantic model
        # No OutputField here - TypedPredictor handles structured output
    
    # Replace docstring with formatted version (DSPy uses __doc__ attribute for prompt generation)
    # This ensures actual relation info is included in the prompt, not placeholders
    RelationExtraction.__doc__ = f"""Extract relationships from text according to the schema.

Extract all relationships between the provided entities based on the schema.
Allowed relations:
{schema_info}
"""
    
    # Store relation info in class for later use (optional, for debugging/validation)
    RelationExtraction._relation_info = schema_info
    
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
        
        # DSPy TypedPredictor kullan (structured output için Pydantic modelleri ile)
        # ChainOfThought yerine TypedPredictor kullan - entity extraction için "thinking" gereksiz
        try:
            # Try TypedPredictor first (DSPy 2.5+)
            if hasattr(dspy, 'TypedPredictor'):
                self.entity_extractor = dspy.TypedPredictor(EntitySig, output_type=EntityList)
                self.relation_extractor = dspy.TypedPredictor(RelationSig, output_type=RelationList)
                self._use_typed_predictor = True
            else:
                # Fallback to Predict if TypedPredictor not available
                logger.warning("TypedPredictor not available, falling back to Predict")
                self.entity_extractor = dspy.Predict(EntitySig)
                self.relation_extractor = dspy.Predict(RelationSig)
                self._use_typed_predictor = False
        except Exception as e:
            # Final fallback to Predict
            logger.warning(f"TypedPredictor initialization failed: {e}, using Predict")
            self.entity_extractor = dspy.Predict(EntitySig)
            self.relation_extractor = dspy.Predict(RelationSig)
            self._use_typed_predictor = False
        
    def forward(self, text: str) -> dspy.Prediction:
        """Extract entities and relations using DSPy TypedPredictor.
        
        Args:
            text: Input text to extract from
        
        Returns:
            dspy.Prediction with entities and relations as lists of tuples
        
        Note:
            - Retry logic is handled by DSPy LM class configuration (max_retries, backoff_factor)
            - dspy.Assert is used for entity type validation (hard constraint - retry if invalid)
            - dspy.Suggest is used for relation validation (soft constraint - hint to LLM)
            - DSPy constraints ensure LLM produces schema-compliant output
        """
        # Step 1: Extract entities
        logger.info("Entity extraction başlatılıyor...")
        
        if self._use_typed_predictor:
            # TypedPredictor returns Pydantic model (EntityList) directly
            # Reference: https://github.com/stanfordnlp/dspy
            entity_result = self.entity_extractor(text=text)
            # entity_result is EntityList, access .entities field
            if isinstance(entity_result, EntityList):
                entities_list = entity_result.entities  # List[Tuple[str, str]]
            else:
                # Fallback: Try to get entities field (should not happen if TypedPredictor works correctly)
                entities_list = getattr(entity_result, 'entities', [])
                logger.warning(f"Expected EntityList, got {type(entity_result).__name__}")
            
            # Ensure it's a list of tuples
            if not isinstance(entities_list, list):
                error_msg = f"Expected list, got {type(entities_list).__name__}"
                logger.error(f"Entity extraction: {error_msg}")
                raise RuntimeError(f"Entity extraction returned invalid type: {error_msg}")
        else:
            # Fallback: Parse from string output (old method - TypedPredictor not available)
            entity_result = self.entity_extractor(text=text)
            entities_str = getattr(entity_result, 'entities', '[]')
            try:
                entities = _parse_json_output(entities_str, expected_format="array")
                entities_list = []
                for item in entities:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        entities_list.append((str(item[0]), str(item[1])))
            except ValueError as e:
                logger.error(f"Entity extraction JSON parsing failed: {e}")
                # Re-raise instead of silently continuing with empty list
                raise RuntimeError(f"Failed to parse entity extraction output: {e}") from e
        
        # DSPy constraint validation: Use dspy.Assert for entity type validation
        # Assert ensures all entity types are valid according to schema
        normalized_schema = _normalize_schema(self.schema)
        valid_entity_types = {e.name for e in normalized_schema.entities}
        
        if entities_list:
            # Filter empty names first (not a DSPy constraint, just data cleaning)
            entities_list = [(name, etype) for name, etype in entities_list if name.strip()]
            
            # Use dspy.Assert to validate entity types (hard constraint)
            # This ensures all entity types are valid according to schema
            # Assert will cause LLM to retry extraction if constraint is violated
            invalid_types = {etype for _, etype in entities_list if etype not in valid_entity_types}
            dspy.Assert(
                len(invalid_types) == 0,
                f"Invalid entity types found: {invalid_types}. "
                f"Valid types are: {sorted(valid_entity_types)}. "
                "Please extract only entities with valid types from the schema."
            )
        
        logger.info(f"Entity extraction tamamlandı: {len(entities_list)} entity bulundu")
        
        # Step 2: Extract relations (entities'i input olarak ver)
        logger.info(f"Relation extraction başlatılıyor ({len(entities_list)} entity ile)...")
        
        if self._use_typed_predictor:
            # TypedPredictor expects List[Tuple] directly, not JSON string
            # Reference: https://github.com/stanfordnlp/dspy
            relation_result = self.relation_extractor(
                text=text,
                entities=entities_list
            )
            # relation_result is RelationList, access .relations field
            if isinstance(relation_result, RelationList):
                relations_list = relation_result.relations  # List[Tuple[str, str, str]]
            else:
                # Fallback: Try to get relations field (should not happen if TypedPredictor works correctly)
                relations_list = getattr(relation_result, 'relations', [])
                logger.warning(f"Expected RelationList, got {type(relation_result).__name__}")
            
            # Ensure it's a list of tuples
            if not isinstance(relations_list, list):
                error_msg = f"Expected list, got {type(relations_list).__name__}"
                logger.error(f"Relation extraction: {error_msg}")
                raise RuntimeError(f"Relation extraction returned invalid type: {error_msg}")
        else:
            # Fallback: Convert entities to JSON string (old method - TypedPredictor not available)
            entities_json = json.dumps(entities_list)
            relation_result = self.relation_extractor(
                text=text,
                entities=entities_json
            )
            relations_str = getattr(relation_result, 'relations', '[]')
            try:
                relations = _parse_json_output(relations_str, expected_format="array")
                relations_list = []
                for item in relations:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        relations_list.append((str(item[0]), str(item[1]), str(item[2])))
            except ValueError as e:
                logger.error(f"Relation extraction JSON parsing failed: {e}")
                # Re-raise instead of silently continuing with empty list
                raise RuntimeError(f"Failed to parse relation extraction output: {e}") from e
        
        # DSPy constraint validation: Use dspy.Suggest for relation validation (soft constraint)
        # Suggest provides hints to LLM about valid relations without forcing retry
        if relations_list and entities_list:
            # Build entity type map
            entity_type_map = {name: etype for name, etype in entities_list}
            entity_names = {name for name, _ in entities_list}
            
            # Get valid relation types from schema
            if isinstance(self.schema, EnhancedDRGSchema):
                valid_relations = set()
                for rg in self.schema.relation_groups:
                    for rel in rg.relations:
                        valid_relations.add((rel.src, rel.name, rel.dst))
            else:
                valid_relations = {(r.src, r.name, r.dst) for r in normalized_schema.relations}
            
            # Use dspy.Assert and dspy.Suggest for relation validation
            # Assert: All relations must reference existing entities (hard constraint)
            missing_refs = [(s, o) for s, _, o in relations_list if s not in entity_names or o not in entity_names]
            if missing_refs:
                dspy.Assert(
                    False,
                    f"Relations reference missing entities: {missing_refs[:5]}. "
                    "All relation entities must be extracted first."
                )
            
            # Suggest: Guide LLM about valid relation types (soft constraint)
            # This provides hints without forcing retry
            invalid_relations = []
            for s, r, o in relations_list:
                s_type = entity_type_map.get(s)
                o_type = entity_type_map.get(o)
                is_valid = s_type and o_type and (s_type, r, o_type) in valid_relations
                if not is_valid:
                    invalid_relations.append((s, r, o, s_type, o_type))
            
            if invalid_relations:
                # Suggest: Some relations may not be valid (soft constraint - hint only)
                invalid_examples = invalid_relations[:3]  # Show first 3 examples
                examples_str = ", ".join([f"{r}({s}:{s_type} -> {o}:{o_type})" for s, r, o, s_type, o_type in invalid_examples])
                dspy.Suggest(
                    len(invalid_relations) == 0,
                    f"Found {len(invalid_relations)} potentially invalid relations (examples: {examples_str}). "
                    f"Consider using valid relation types from schema: {sorted(set(r for _, r, _, _, _ in invalid_relations))}"
                )
        
        logger.info(f"Relation extraction tamamlandı: {len(relations_list)} relation bulundu")
        
<<<<<<< HEAD
        # Build enriched relations with metadata
        enriched_relations = []
        for i, rel in enumerate(relations_list):
            rel_dict = {
                "relation": rel,
                "confidence": confidence_scores[i] if confidence_scores else None,
                "temporal": temporal_info[i] if temporal_info else None,
                "is_negated": negations[i] if negations is not None else False,
            }
            enriched_relations.append(rel_dict)
        
        if _should_return_dspy_prediction():
            return dspy.Prediction(
                entities=entities_list,
                relations=relations_list,
                enriched_relations=enriched_relations,
            )
        return ExtractionResult(
            entities=entities_list,
            relations=relations_list,
            enriched_relations=enriched_relations,
        )


def _infer_relation_metadata_heuristic(
    text: str,
    relations: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """Heuristic relation metadata inference (English-first, conservative).
    
    Only used when the LLM did not provide temporal/negation metadata.
    We prefer abstaining (None/False) over guessing.
    """
    if not text or not relations:
        return {"temporal_info": None, "negations": None}

    lang = os.getenv("DRG_LANGUAGE", "en").lower()
    if lang not in {"en", "english"}:
        return {"temporal_info": None, "negations": None}

    temporal_info: List[Optional[Dict[str, Optional[str]]]] = []
    negations: List[bool] = []

    for s, r, o in relations:
        window = _find_evidence_window(text, s, o, window_chars=220)
        neg = _detect_negation_in_window(window, relation_name=r)
        temporal = _extract_year_temporal(window)
        temporal_info.append(temporal)
        negations.append(neg)

    return {"temporal_info": temporal_info, "negations": negations}


def _find_evidence_window(text: str, a: str, b: str, window_chars: int = 200) -> str:
    """Find a short window around the closest mentions of a and b (word-boundary)."""
    if not text or not a or not b:
        return ""

    def _positions(term: str) -> List[int]:
        pattern = rf"(?i)(?<!\w){re.escape(term)}(?!\w)"
        return [m.start() for m in re.finditer(pattern, text)]

    pos_a = _positions(a)
    pos_b = _positions(b)
    if not pos_a or not pos_b:
        return ""

    best_pair = None
    best_dist = None
    for pa in pos_a:
        for pb in pos_b:
            d = abs(pa - pb)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pair = (pa, pb)

    if best_pair is None:
        return ""

    # Prefer the sentence containing the closest pair (reduces false negation bleed from adjacent sentences).
    center = int((best_pair[0] + best_pair[1]) / 2)
    left_punct = max(text.rfind(".", 0, center), text.rfind("?", 0, center), text.rfind("!", 0, center))
    sent_start = 0 if left_punct == -1 else left_punct + 1
    right_candidates = [p for p in (text.find(".", center), text.find("?", center), text.find("!", center)) if p != -1]
    sent_end = (min(right_candidates) + 1) if right_candidates else len(text)
    sent = text[sent_start:sent_end].strip()
    if sent:
        return sent

    left = max(0, min(best_pair) - window_chars)
    right = min(len(text), max(best_pair) + window_chars)
    return text[left:right]


def _detect_negation_in_window(window: str, relation_name: str) -> bool:
    """Detect obvious negation cues near a relation mention (conservative)."""
    if not window:
        return False

    w = window.lower()
    cues = [
        "no longer",
        "never",
        "did not",
        "does not",
        "do not",
        "is not",
        "was not",
        "are not",
        "were not",
        "cannot",
        "can't",
        "won't",
        "ceased to",
        "stopped",
        "discontinued",
        "discontinue",
    ]
    if not any(c in w for c in cues):
        return False

    # Relation-token check (stem) to avoid tagging unrelated negations.
    stem = (relation_name or "").split("_")[0].lower()
    if stem and len(stem) >= 4 and stem in w:
        return True

    # Special-case common production verbs (allow inflections).
    if any(k in w for k in ["produc", "manufactur", "mak"]):
        if stem in {"produces", "produce", "produced", "manufactures", "manufacture", "makes", "make"}:
            return True

    return False


def _extract_year_temporal(window: str) -> Optional[Dict[str, Optional[str]]]:
    """Extract simple year-only temporal metadata from a window.
    
    Returns:
        {"start": "YYYY-01-01", "end": "YYYY-12-31"} for explicit ranges,
        {"start": "YYYY-01-01", "end": None} for single-year mentions,
        or None if no year detected.
    """
    if not window:
        return None
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", window)]
    if not years:
        return None
    years = sorted(set(years))

    m = re.search(
        r"\b(19\d{2}|20\d{2})\b\s*(?:-|to|until|through)\s*\b(19\d{2}|20\d{2})\b",
        window,
        flags=re.IGNORECASE,
    )
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        return {"start": f"{min(y1, y2)}-01-01", "end": f"{max(y1, y2)}-12-31"}

    return {"start": f"{years[0]}-01-01", "end": None}
=======
        # Return DSPy Prediction (standard return type for DSPy Modules)
        # Note: Manual Prediction creation is acceptable for multi-step modules like KGExtractor
        # that combine results from multiple predictors (entity extraction + relation extraction)
        return dspy.Prediction(
            entities=entities_list,
            relations=relations_list
        )
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce


# Global extractor instance (lazy initialized)
_extractor: Optional[KGExtractor] = None


def _configure_llm_auto():
    """DSPy LLM'ini otomatik olarak environment variable'lardan konfigüre et.
    
    This function now uses the LMConfig class from drg.config module
    for better testability and separation of concerns.
    """
    # Try importing configure_lm (lazy import pattern)
    try:
        from .config import configure_lm
        configure_lm()
        return
    except ImportError:
        logger.warning("drg.config module not available, skipping LM configuration")
        return


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


def extract_typed(
    text: str,
    schema: Union[DRGSchema, EnhancedDRGSchema],
    enable_entity_resolution: bool = True,
    enable_coreference_resolution: bool = False,
<<<<<<< HEAD
    enable_implicit_relationships: bool = True,
    enable_cross_chunk_context_snippets: bool = True,
    max_cross_chunk_context_chunks: int = 3,
    cross_chunk_snippet_chars: int = 350,
    max_cross_chunk_context_chars: int = 1200,
    min_anchor_entity_len: int = 3,
    max_anchor_entities: int = 8,
        two_pass_extraction: bool = True,  # Default True for better cross-chunk relationship discovery
    embedding_provider=None,  # Optional embedding provider for entity/coreference resolution
=======
    use_optimizer: bool = False,
    optimizer_config: Optional[Any] = None,
    training_examples: Optional[List[Dict[str, Any]]] = None,
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations from text using DSPy.
    
    Tamamen declarative - sadece schema tanımlıyorsun, DSPy gerisini hallediyor.
    
    Args:
        text: Input text to extract from
        schema: DRGSchema defining allowed entity types and relations
        enable_entity_resolution: Whether to enable entity resolution (merges duplicate entity names)
        enable_coreference_resolution: Whether to enable coreference resolution (resolves pronouns/references to entities)
        use_optimizer: Whether to use optimized extractor (requires training_examples)
        optimizer_config: Optional OptimizerConfig for custom optimizer settings
        training_examples: Optional list of training examples for optimizer. Format:
            [{"text": str, "expected_entities": List[Tuple[str, str]], "expected_relations": List[Tuple[str, str, str]]}, ...]
    
    Returns:
        Tuple of (entities_typed, triples) where:
        - entities_typed: List of (entity_name, entity_type) tuples
        - triples: List of (source, relation, target) tuples
    """
<<<<<<< HEAD
    extractor = _get_extractor(schema)
    
    if two_pass_extraction:
        # Two-pass extraction: better for cross-chunk relationships
        logger.info("Using two-pass extraction mode")
        
        # PASS 1: Extract ALL entities from ALL chunks
        logger.info("Pass 1: Extracting entities from all chunks...")
        all_entities = []
        chunk_texts = []  # Store chunk texts for pass 2
        chunk_entities_list: List[List[Tuple[str, str]]] = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
            if not chunk_text.strip():
                chunk_texts.append("")
                chunk_entities_list.append([])
                continue
            
            chunk_texts.append(chunk_text)
            logger.info(f"Pass 1 - Processing chunk {i+1}/{len(chunks)} for entity extraction...")
            
            # Extract entities only (no relations in pass 1)
            # Use forward() but we'll only use entities
            throttle_llm_calls()
            result = extractor(text=chunk_text)
            chunk_entities = result.entities if hasattr(result, 'entities') else []
            chunk_entities_list.append(chunk_entities)
            
            # Deduplicate entities as we collect them
            existing_entities = {(name.lower(), etype) for name, etype in all_entities}
            for name, etype in chunk_entities:
                if (name.lower(), etype) not in existing_entities:
                    all_entities.append((name, etype))
                    existing_entities.add((name.lower(), etype))
        
        logger.info(f"Pass 1 complete: {len(all_entities)} unique entities extracted")

        # Build entity -> chunk indices map for deterministic cross-chunk context snippets.
        entity_to_chunks: Dict[str, List[int]] = {}
        for idx, ents in enumerate(chunk_entities_list):
            for name, _ in ents:
                if not name:
                    continue
                key = name.lower()
                entity_to_chunks.setdefault(key, [])
                if not entity_to_chunks[key] or entity_to_chunks[key][-1] != idx:
                    entity_to_chunks[key].append(idx)
        
        # PASS 2: Extract relations with GLOBAL entity context
        logger.info(f"Pass 2: Extracting relations with {len(all_entities)} global entities...")
        all_triples = []
        
        for i, chunk_text in enumerate(chunk_texts):
            if not chunk_text.strip():
                continue
            
            logger.info(f"Pass 2 - Processing chunk {i+1}/{len(chunks)} for relation extraction...")
            
            # Deterministic intra-document evidence injection: add short excerpts from other chunks that share entities
            # with this chunk. This is NOT retrieval/RAG; it only reuses already-ingested document chunks.
            augmented_text = chunk_text
            if (
                enable_cross_chunk_relationships
                and enable_cross_chunk_context_snippets
                and max_cross_chunk_context_chunks > 0
            ):
                current_entities = _select_anchor_entities(
                    chunk_text=chunk_text,
                    chunk_entities=chunk_entities_list[i],
                    entity_to_chunks=entity_to_chunks,
                    total_chunks=len(chunk_texts),
                    min_anchor_len=min_anchor_entity_len,
                    max_anchors=max_anchor_entities,
                )
                snippets = _build_cross_chunk_context_snippets(
                    chunk_texts=chunk_texts,
                    entity_to_chunks=entity_to_chunks,
                    anchor_entities=current_entities,
                    current_chunk_index=i,
                    max_chunks=max_cross_chunk_context_chunks,
                    snippet_chars=cross_chunk_snippet_chars,
                    max_total_chars=max_cross_chunk_context_chars,
                    min_anchor_len=min_anchor_entity_len,
                )
                if snippets:
                    augmented_text = (
                        "[CROSS-CHUNK CONTEXT]\n"
                        + "\n\n".join(snippets)
                        + "\n\n[CURRENT CHUNK]\n"
                        + chunk_text
                    )

            if enable_cross_chunk_relationships:
                throttle_llm_calls()
                result = extractor(text=augmented_text, context_entities=all_entities)
            else:
                throttle_llm_calls()
                result = extractor(text=chunk_text)
            
            chunk_relations = result.relations if hasattr(result, 'relations') else []
            all_triples.extend(chunk_relations)
        
        logger.info(f"Pass 2 complete: {len(all_triples)} relations extracted")
        
    else:
        # Single-pass extraction: original incremental context approach
        logger.info("Using single-pass extraction mode")
        all_entities = []
        all_triples = []
        context_entities = []  # Entities from previous chunks
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
            if not chunk_text.strip():
                continue
            
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Extract with context entities if enabled
            if enable_cross_chunk_relationships and context_entities:
                throttle_llm_calls()
                result = extractor(text=chunk_text, context_entities=context_entities)
            else:
                throttle_llm_calls()
                result = extractor(text=chunk_text)
            
            chunk_entities = result.entities if hasattr(result, 'entities') else []
            chunk_relations = result.relations if hasattr(result, 'relations') else []
            
            # Add to results
            all_entities.extend(chunk_entities)
            all_triples.extend(chunk_relations)
            
            # Update context entities (accumulate entities from ALL previous chunks for cross-chunk relationships)
            if enable_cross_chunk_relationships:
                # Merge with context, avoiding duplicates
                # This accumulates entities from ALL previous chunks, not just recent ones
                # This allows chunk 15 to see entities from chunk 1, enabling cross-chunk relationship discovery
                existing_names = {(name.lower(), etype) for name, etype in context_entities}
                for name, etype in chunk_entities:
                    if (name.lower(), etype) not in existing_names:
                        context_entities.append((name, etype))
                
                logger.info(
                    f"Context entities updated: {len(context_entities)} total entities "
                    f"(chunk {i+1} can see entities from chunks 1-{i+1})"
                )
    
    # Deduplicate entities
    entity_set = set(all_entities)
    all_entities = list(entity_set)
    
    # Deduplicate relations
    relation_set = set(all_triples)
    all_triples = list(relation_set)
    
    # Apply post-processing
    if enable_coreference_resolution:
        if resolve_coreferences:
            try:
                # Combine all chunk texts for coreference resolution
                full_text = "\n\n".join([chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in chunks])
                all_entities, all_triples = resolve_coreferences(
                    text=full_text,
                    entities=all_entities,
                    relations=all_triples,
                    use_nlp=True,
                    use_neural_coref=True,
                    embedding_provider=embedding_provider,  # Pass embedding provider for semantic similarity-based disambiguation
                    language=os.getenv("DRG_LANGUAGE", "en"),
                )
            except Exception as e:
                logger.warning(f"Coreference resolution failed: {e}")
    
    if enable_entity_resolution:
        if resolve_entities_and_relations:
            try:
                all_entities, all_triples = resolve_entities_and_relations(
                    all_entities,
                    all_triples,
                    similarity_threshold=0.65,  # Lowered from 0.85 for better recall
                    adaptive_threshold=True,  # Adaptive threshold for short names
                    embedding_provider=embedding_provider,  # Pass embedding provider for better entity resolution
                    use_embedding=True  # Use embedding-based similarity if available
                )
            except Exception as e:
                logger.warning(f"Entity resolution failed: {e}")
    
    # Optional: deterministic implicit relationship inference (schema-gated) on the concatenated text.
    if enable_implicit_relationships and all_entities:
        try:
            full_text = "\n\n".join(
                [chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in chunks]
            )
            inferred = _infer_implicit_relations(
                text=full_text, entities=all_entities, schema=schema, existing_triples=all_triples
            )
            if inferred:
                existing = set(all_triples)
                for t in inferred:
                    if t not in existing:
                        all_triples.append(t)
                        existing.add(t)
        except Exception as e:
            logger.debug(f"Implicit relationship inference failed: {e}")
    
    # Validation: Check for hub entities (configurable)
    #
    # Rationale: Some documents are naturally "hubby" (many facts around a single central entity).
    # This validation is useful as a quality gate in some experiments, but it must NOT hard-fail
    # all pipelines by default. Therefore it's configurable via env:
    # - DRG_VALIDATE_HUB_DOMINANCE: 1/0 (default: 1)
    # - DRG_HUB_VALIDATION_MODE: "error" | "warn" (default: "error")
    # - DRG_MAX_HUB_RATIO: float (default: 0.30)
    # - DRG_MIN_DIVERSITY_RATIO: float (default: 0.50)
    from collections import Counter
    entity_counts = Counter()
    for s, r, t in all_triples:
        entity_counts[s] += 1
        entity_counts[t] += 1
    
    total_edges = len(all_triples)
    # Default OFF: This project is KG extraction-focused; hub-ness can be a natural property of many texts.
    # Keep it as an *optional* QA/experiment gate only.
    validate_hub = os.getenv("DRG_VALIDATE_HUB_DOMINANCE", "0").strip().lower() in {"1", "true", "yes", "y"}
    validation_mode = os.getenv("DRG_HUB_VALIDATION_MODE", "error").strip().lower()
    if total_edges > 0 and validate_hub:
        try:
            max_hub_ratio = float(os.getenv("DRG_MAX_HUB_RATIO", "0.30"))
        except Exception:
            max_hub_ratio = 0.30
        try:
            min_diversity_ratio = float(os.getenv("DRG_MIN_DIVERSITY_RATIO", "0.50"))
        except Exception:
            min_diversity_ratio = 0.50  # At least half of relations must NOT involve the top entity
        hub_entities = entity_counts.most_common(1)
        top_entity = hub_entities[0][0] if hub_entities else None
        top_count = hub_entities[0][1] if hub_entities else 0
        hub_ratio = top_count / total_edges if total_edges else 0.0
        diversity_ratio = (total_edges - top_count) / total_edges if total_edges else 0.0
        
        if hub_ratio > max_hub_ratio or diversity_ratio < min_diversity_ratio:
            msg = (
                "Hub dominance validation failed: "
                f"top_entity={top_entity}, hub_ratio={hub_ratio:.2f} "
                f"(max={max_hub_ratio:.2f}), diversity_ratio={diversity_ratio:.2f} "
                f"(min={min_diversity_ratio:.2f})."
            )
            if validation_mode == "warn":
                logger.warning(f"⚠️  {msg} Proceeding (mode=warn).")
            else:
                logger.error(f"❌ {msg}")
                logger.error(
                    "   If this document is naturally hub-like, disable this gate via "
                    "DRG_VALIDATE_HUB_DOMINANCE=0 or set DRG_HUB_VALIDATION_MODE=warn."
                )
            raise ValueError("Hub dominance validation failed")
    
    return all_entities, all_triples


def _build_cross_chunk_context_snippets(
    chunk_texts: List[str],
    entity_to_chunks: Dict[str, List[int]],
    anchor_entities: List[str],
    current_chunk_index: int,
    max_chunks: int,
    snippet_chars: int,
    max_total_chars: int = 1200,
    min_anchor_len: int = 3,
) -> List[str]:
    """Build short evidence snippets from other chunks sharing anchor entity mentions.
    
    This is deterministic, string-indexed intra-document evidence selection (not retrieval/RAG).
    """
    if not chunk_texts or not anchor_entities or max_chunks <= 0:
        return []

    # Filter anchors: avoid very short/ambiguous terms and common pronouns.
    pronoun_like = {
        "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs",
        "o", "onlar", "ona", "onu", "onun", "onların", "bu", "şu", "that", "this",
    }
    filtered_anchors: List[str] = []
    for ent in anchor_entities:
        if not ent:
            continue
        ent_s = ent.strip()
        if len(ent_s) < min_anchor_len:
            continue
        if ent_s.lower() in pronoun_like:
            continue
        # Skip pure numbers
        if ent_s.isdigit():
            continue
        filtered_anchors.append(ent_s)
    if not filtered_anchors:
        return []

    def _contains_entity(text: str, entity: str) -> bool:
        # Word-boundary match to reduce substring collisions ("us" in "business").
        # Accepts unicode word chars.
        pattern = r"(?i)(?<!\\w)" + re.escape(entity) + r"(?!\\w)"
        return re.search(pattern, text) is not None
    # Collect candidate chunk indices that contain any anchor entity (excluding current).
    candidates: Set[int] = set()
    for ent in filtered_anchors:
        idxs = entity_to_chunks.get(ent.lower(), [])
        for j in idxs:
            if j != current_chunk_index:
                candidates.add(j)
    if not candidates:
        return []

    # Rank by proximity (closer chunks first), then by index for stability.
    # Rank: (how many anchors this chunk contains desc, proximity asc, index asc)
    def _anchor_hits(j: int) -> int:
        t = chunk_texts[j] or ""
        return sum(1 for a in filtered_anchors if _contains_entity(t, a))
    ranked = sorted(
        candidates,
        key=lambda j: (-_anchor_hits(j), abs(j - current_chunk_index), j),
    )
    ranked = [j for j in ranked if _anchor_hits(j) > 0][:max_chunks]

    def _snippet_for(text: str, term: str) -> str:
        if not term:
            return text[:snippet_chars].strip()
        # Find first word-boundary match position
        pattern = r"(?i)(?<!\\w)" + re.escape(term) + r"(?!\\w)"
        m = re.search(pattern, text)
        if not m:
            return text[:snippet_chars].strip()
        pos = m.start()
        start = max(0, pos - snippet_chars // 2)
        end = min(len(text), start + snippet_chars)
        return text[start:end].strip()

    snippets: List[str] = []
    # Use first anchor that exists in each candidate chunk to center the snippet.
    for j in ranked:
        t = chunk_texts[j]
        if not t:
            continue
        chosen = None
        for ent in filtered_anchors:
            if _contains_entity(t, ent):
                chosen = ent
                break
        excerpt = _snippet_for(t, chosen) if chosen else t[:snippet_chars].strip()
        snippets.append(f"Chunk {j+1} excerpt: {excerpt}")

    # Enforce total context budget.
    if max_total_chars > 0 and snippets:
        out: List[str] = []
        total = 0
        for s in snippets:
            if total + len(s) > max_total_chars:
                break
            out.append(s)
            total += len(s)
        return out
    return snippets


def _select_anchor_entities(
    chunk_text: str,
    chunk_entities: List[Tuple[str, str]],
    entity_to_chunks: Dict[str, List[int]],
    total_chunks: int,
    min_anchor_len: int,
    max_anchors: int,
) -> List[str]:
    """Select a small, robust set of anchor entities for cross-chunk evidence injection.
    
    Uses TF-IDF-like scoring:
      score = tf_in_current_chunk * (log((N+1)/(df+1)) + 1)
    This downweights entities that appear in many chunks (too generic) and prioritizes
    entities that are salient in the current chunk.
    """
    if not chunk_text or not chunk_entities or max_anchors <= 0:
        return []
    text = chunk_text

    pronoun_like = {
        "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs",
        "o", "onlar", "ona", "onu", "onun", "onların", "bu", "şu", "that", "this",
    }

    def _count_occurrences(term: str) -> int:
        if not term:
            return 0
        # Word-boundary match, case-insensitive.
        pattern = r"(?i)(?<!\\w)" + re.escape(term) + r"(?!\\w)"
        return len(re.findall(pattern, text))

    scored: List[Tuple[float, str]] = []
    seen = set()
    for name, _ in chunk_entities:
        if not name:
            continue
        term = name.strip()
        if term.lower() in seen:
            continue
        seen.add(term.lower())
        if len(term) < min_anchor_len:
            continue
        if term.lower() in pronoun_like:
            continue
        if term.isdigit():
            continue
        tf = _count_occurrences(term)
        if tf <= 0:
            continue
        df = len(entity_to_chunks.get(term.lower(), []))
        idf = math.log((total_chunks + 1) / (df + 1)) + 1.0
        score = tf * idf * (1.0 + min(len(term) / 20.0, 1.0) * 0.1)  # slight preference for longer names
        scored.append((score, term))

    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    return [t for _, t in scored[:max_anchors]]


def extract_typed(
    text: str,
    schema: Union[DRGSchema, EnhancedDRGSchema],
    enable_entity_resolution: bool = True,
    enable_coreference_resolution: bool = False,
    enable_implicit_relationships: bool = True,
    embedding_provider=None,
    use_optimizer: bool = False,
    optimizer_config: Optional[Any] = None,
    training_examples: Optional[List[Dict[str, Any]]] = None,
        return_enriched: bool = False,
        min_confidence: Optional[float] = None,
) -> Union[
    Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]],
    Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]], List[Dict[str, Any]]]
]:
    """
    Extract entities and relations from text using DSPy.
    
    Tamamen declarative - sadece schema tanımlıyorsun, DSPy gerisini hallediyor.
    
    **IMPORTANT**: This function processes text as a SINGLE chunk. If your text is long and needs chunking,
    use `extract_from_chunks()` instead to enable cross-chunk relationship discovery. This function will
    miss relationships between entities that are far apart in the text if it exceeds LLM context window.
    
    For multi-chunk texts, use:
        from drg.chunking import create_chunker
        from drg.extract import extract_from_chunks
        
        chunker = create_chunker(strategy="token_based", chunk_size=768, overlap_ratio=0.15)
        chunks = chunker.chunk(text)
        entities, relations = extract_from_chunks(
            chunks=[{"text": chunk.text} for chunk in chunks],
            schema=schema,
            enable_cross_chunk_relationships=True  # IMPORTANT: Enables cross-chunk relationship discovery
        )
    
    Args:
        text: Input text to extract from (processed as single chunk)
        schema: DRGSchema defining allowed entity types and relations
        enable_entity_resolution: Whether to enable entity resolution (merges duplicate entity names)
        enable_coreference_resolution: Whether to enable coreference resolution (resolves pronouns/references to entities)
        enable_implicit_relationships: Whether to infer a small set of implicit relations from surface patterns
                                     (e.g., possessives) in a schema-gated way. Default: True.
        embedding_provider: Optional embedding provider for semantic similarity-based disambiguation in
                           coreference resolution and entity resolution. Default: None.
        use_optimizer: Whether to use optimized extractor (requires training_examples)
        optimizer_config: Optional OptimizerConfig for custom optimizer settings
        training_examples: Optional list of training examples for optimizer. Format:
            [{"text": str, "expected_entities": List[Tuple[str, str]], "expected_relations": List[Tuple[str, str, str]]}, ...]
        return_enriched: If True, returns enriched relations metadata (temporal info, confidence, negation). 
                        Default: False for backward compatibility.
        min_confidence: Optional minimum confidence threshold (0.0-1.0) to filter relationships.
                       Relationships with confidence < min_confidence will be filtered out.
                       Only applies when enriched_relations are available and confidence scores are provided.
                       Default: None (no filtering). Recommended: 0.5-0.7 for quality control.
    
    Returns:
        If return_enriched=False (default):
            Tuple of (entities_typed, triples) where:
            - entities_typed: List of (entity_name, entity_type) tuples
            - triples: List of (source, relation, target) tuples
        
        If return_enriched=True:
            Tuple of (entities_typed, triples, enriched_relations) where:
            - entities_typed: List of (entity_name, entity_type) tuples
            - triples: List of (source, relation, target) tuples (filtered by min_confidence if set)
            - enriched_relations: List of dicts with metadata for each relation:
                [{"relation": (s, r, t), "confidence": float, "temporal": {"start": str, "end": str}, "is_negated": bool}, ...]
    """
    # Fast paths / mock mode support
    if not text or not text.strip():
        if return_enriched:
            return [], [], []
        return [], []

=======
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
    # Get base extractor
    extractor = _get_extractor(schema)
    
    # Use optimizer if requested and training examples provided
    if use_optimizer and training_examples:
        if DRGOptimizer is None or OptimizerConfig is None:
            logger.warning("Optimizer module not available, falling back to base extractor")
        else:
            try:
                # Use provided config or default
                if optimizer_config is None:
                    optimizer_config = OptimizerConfig()
                
                # Create optimizer with training examples (can be passed to constructor)
                optimizer = DRGOptimizer(
                    schema=schema,
                    config=optimizer_config,
                    training_examples=training_examples  # Pass directly to constructor
                )
                
                # Optimize extractor
                logger.info(f"Optimizing extractor with {len(training_examples)} training examples...")
                extractor = optimizer.optimize()  # This will raise RuntimeError if optimization fails
                logger.info("Optimization completed, using optimized extractor")
            except (RuntimeError, Exception) as e:
                # optimizer.optimize() raises RuntimeError on failure (no silent fallback)
                # Re-raise to inform user that optimization was requested but failed
                logger.error(f"Optimizer failed: {e}")
                raise RuntimeError(
                    f"Optimizer optimization failed: {e}. "
                    "If you want to proceed without optimization, set use_optimizer=False. "
                    "Otherwise, check your training examples format and optimizer configuration."
                ) from e
    
    result = extractor(text=text)
    
    # KGExtractor artık zaten list of tuples döndürüyor
    entities_typed = result.entities if hasattr(result, 'entities') and result.entities else []
    triples = result.relations if hasattr(result, 'relations') and result.relations else []
    
    # Validate types (should not happen if KGExtractor works correctly)
    if not isinstance(entities_typed, list):
        error_msg = f"Expected list for entities, got {type(entities_typed).__name__}"
        logger.error(f"Extract typed: {error_msg}")
        raise RuntimeError(f"Extraction returned invalid entities type: {error_msg}")
    if not isinstance(triples, list):
        error_msg = f"Expected list for triples, got {type(triples).__name__}"
        logger.error(f"Extract typed: {error_msg}")
        raise RuntimeError(f"Extraction returned invalid triples type: {error_msg}")
    
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
    
    # Coreference Resolution: Resolve pronouns and references to explicit entities
    # This should be applied BEFORE entity resolution, as it creates explicit mentions
    # Example: "Elon Musk founded Tesla. He is the CEO." → "He" → "Elon Musk"
    if enable_coreference_resolution and valid_entities:
        if resolve_coreferences is None:
            logger.warning("Coreference resolution module not available, skipping resolution")
        else:
            try:
                valid_entities, valid_triples = resolve_coreferences(
                    text=text,
                    entities=valid_entities,
                    relations=valid_triples,
                    use_nlp=True  # Use NLP if available, falls back to heuristics otherwise
                )
                logger.info("Coreference resolution applied successfully")
            except Exception as e:
                logger.warning(f"Coreference resolution failed: {e}, continuing without resolution")
    
    # Entity Resolution: Merge duplicate entity references
    # This is critical for KG quality - same entity appearing with different names
    # (e.g., "Dr. Elena Vasquez", "Dr. Vasquez", "Dr. Elena") should be merged
    # Applied AFTER coreference resolution to merge all mentions (including resolved pronouns)
    if enable_entity_resolution and valid_entities:
        if resolve_entities_and_relations is None:
            logger.warning("Entity resolution module not available, skipping resolution")
        else:
            try:
                valid_entities, valid_triples = resolve_entities_and_relations(
                    valid_entities,
                    valid_triples,
                    similarity_threshold=0.85
                )
                logger.info("Entity resolution applied successfully")
            except Exception as e:
                logger.warning(f"Entity resolution failed: {e}, continuing without resolution")
    
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
    
    NOTE: This function uses a detailed prompt (112 lines) in SchemaGeneration signature,
    which is against DSPy's typical "declarative" philosophy of minimal prompts.
    However, schema generation is a complex task requiring explicit instructions to produce
    rich, interconnected schemas. The detailed prompt is intentional and necessary for
    generating high-quality schemas with diverse relation patterns.
    
    For standard extraction tasks, use KGExtractor which follows DSPy's declarative approach
    with minimal signatures and TypedPredictor for structured output.
    
    Args:
        text: Analiz edilecek metin
        max_retries: Rate limit hatası durumunda maksimum deneme sayısı (DEPRECATED - retry handled by DSPy LM)
        retry_delay: Denemeler arası bekleme süresi (saniye) (DEPRECATED - retry handled by DSPy LM)
    
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
    # NOTE: ChainOfThought is used here because schema generation requires complex reasoning
    # to create rich, interconnected schemas. For entity/relation extraction, TypedPredictor is preferred.
    schema_generator = dspy.ChainOfThought(SchemaGeneration)
    
    # Şema oluştur (retry logic DSPy LM class tarafından handle ediliyor)
    # NOTE: Manual retry logic removed - DSPy LM class handles retries automatically via
    # max_retries and backoff_factor parameters configured in drg.config.configure_lm()
    try:
<<<<<<< HEAD
        if hasattr(dspy, 'TypedPredictor'):
            schema_generator = dspy.TypedPredictor(SchemaGeneration, output_type=SchemaOutput)
            throttle_llm_calls()
            schema_result = schema_generator(text=sample_text)
            # TypedPredictor returns Pydantic model directly
            if isinstance(schema_result, SchemaOutput):
                schema_str = schema_result.generated_schema
            else:
                schema_str = getattr(schema_result, 'generated_schema', '{}')
        else:
            # Fallback to ChainOfThought if TypedPredictor not available
            schema_generator = dspy.ChainOfThought(SchemaGeneration)
            throttle_llm_calls()
            schema_result = schema_generator(text=sample_text)
            schema_str = schema_result.generated_schema if hasattr(schema_result, 'generated_schema') else "{}"
        
=======
        schema_result = schema_generator(text=sample_text)
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
        logger.info("Schema generation tamamlandı")
    except Exception as e:
        logger.error(f"Schema generation failed: {e}")
        raise RuntimeError(f"Schema generation failed: {e}. Check your LLM configuration and API keys.") from e
    
    # Parse JSON schema
    schema_str = schema_result.generated_schema if hasattr(schema_result, 'generated_schema') else "{}"
    try:
        schema_data = _parse_json_output(schema_str, expected_format="object")
    except ValueError as e:
        logger.error(f"Failed to parse schema JSON: {e}")
<<<<<<< HEAD
        logger.error(f"Raw schema output (first 1000 chars): {schema_str[:1000]}")
        raise RuntimeError(
            f"Schema JSON parsing failed: {e}. "
            "This usually means the LLM output format is incorrect. "
            "Check your LLM configuration or try a different model."
        ) from e
=======
        logger.warning("Schema parsing failed, using default schema")
        return _create_default_enhanced_schema()
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
    
    # Validate schema_data is not empty
    if not schema_data or (isinstance(schema_data, dict) and not schema_data):
        logger.error("Parsed schema JSON is empty, using default schema")
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
<<<<<<< HEAD
                relation_groups=[
                    RelationGroup(
                        name="general",
                        description="General relations",
                    relations=relations,
                    )
                ],
                auto_discovery=bool(schema_data.get("auto_discovery", False)),
            )
        else:
            raise RuntimeError(
                f"Schema generation output is not a valid EnhancedDRGSchema JSON: {e}"
            ) from e
    
    # Add reverse relations automatically for bidirectional extraction support.
    schema = EnhancedDRGSchema(
        entity_types=schema.entity_types,
        relation_groups=_add_reverse_relations(schema.relation_groups, schema.entity_types),
        auto_discovery=schema.auto_discovery,
    )

    total_relations_count = sum(len(rg.relations) for rg in schema.relation_groups)
    logger.info(
        f"Enhanced schema oluşturuldu: {len(schema.entity_types)} entity type, "
        f"{len(schema.relation_groups)} relation group, {total_relations_count} relation"
    )
    return schema
=======
                relation_groups=valid_relation_groups,
                auto_discovery=True
            )
        else:
            # Son çare: varsayılan enhanced şema
            logger.warning("Hatalı şema, varsayılan enhanced şema kullanılıyor...")
            return _create_default_enhanced_schema()
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce


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
    
<<<<<<< HEAD
    Goals:
    - Keep behavior deterministic (no randomness).
    - Maximize document coverage for long inputs (avoid missing late-section types/relations).
    - Enforce a strict budget to stay safe for any input size.
    """
    if not text or not text.strip():
        return ""

    target_coverage = float(os.getenv("DRG_SCHEMA_TARGET_COVERAGE", "0.60"))
    max_total_chars = int(os.getenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "100000"))
    max_parts = int(os.getenv("DRG_SCHEMA_MAX_PARTS", "20"))
    min_part_chars = int(os.getenv("DRG_SCHEMA_MIN_PART_CHARS", "2500"))
    max_part_chars = int(os.getenv("DRG_SCHEMA_MAX_PART_CHARS", "5000"))

    doc_len = len(text)
    if doc_len <= max_total_chars:
        logger.info(f"Metin kısa/orta ({doc_len:,} karakter), tamamı kullanılıyor...")
        return text

    desired = min(max_total_chars, max(min_part_chars * 4, int(doc_len * max(0.0, min(1.0, target_coverage)))))
    # Choose number of parts based on desired size and a reasonable per-part budget.
    per_part = max(min_part_chars, min(max_part_chars, desired // max(4, min(max_parts, 12))))
    num_parts = max(4, min(max_parts, max(4, int(round(desired / max(1, per_part))))))
    part_len = max(min_part_chars, min(max_part_chars, desired // num_parts))

    all_parts: List[Tuple[int, str]] = []
    for i in range(num_parts):
        if i == 0:
            start = 0
        elif i == num_parts - 1:
            start = max(0, doc_len - part_len)
        else:
            ratio = i / (num_parts - 1)
            center = int(doc_len * ratio)
            start = max(0, center - part_len // 2)
        end = min(start + part_len, doc_len)
        if end > start:
            all_parts.append((i, text[start:end]))

    # Budget enforcement (defensive):
    # Always include FIRST and LAST parts, then fill from the middle out until budget is exhausted.
    sep = "\n\n[... truncated ...]\n\n"
    parts_by_idx = {i: p for i, p in all_parts}
    chosen_idxs: List[int] = []
    if 0 in parts_by_idx:
        chosen_idxs.append(0)
    last_idx = num_parts - 1
    if last_idx in parts_by_idx and last_idx not in chosen_idxs:
        chosen_idxs.append(last_idx)

    # Candidate fill order: spread coverage by alternating around the middle.
    mids = [i for i in range(1, last_idx) if i in parts_by_idx]
    mid_center = (last_idx) / 2.0
    mids.sort(key=lambda i: (abs(i - mid_center), i))

    def _serialized_len(idxs: List[int]) -> int:
        if not idxs:
            return 0
        return sum(len(parts_by_idx[i]) for i in idxs) + (len(idxs) - 1) * len(sep)

    for i in mids:
        trial = chosen_idxs + [i]
        trial_sorted = sorted(set(trial))
        if _serialized_len(trial_sorted) <= max_total_chars:
            chosen_idxs = trial_sorted

    # If last part is missing (shouldn't happen), force it by dropping mids.
    if last_idx in parts_by_idx and last_idx not in chosen_idxs:
        chosen_idxs.append(last_idx)
        chosen_idxs = sorted(set(chosen_idxs))
        # Drop from the middle until we fit.
        while _serialized_len(chosen_idxs) > max_total_chars and len(chosen_idxs) > 2:
            # remove the index closest to the middle first
            removable = [i for i in chosen_idxs if i not in (0, last_idx)]
            if not removable:
                break
            removable.sort(key=lambda i: (abs(i - mid_center), i), reverse=True)
            chosen_idxs.remove(removable[0])

    out_parts = [parts_by_idx[i] for i in chosen_idxs]

    sampled = sep.join(out_parts)
    coverage = (sum(len(p) for p in out_parts) / doc_len) * 100.0
    logger.info(
        f"Metin çok uzun ({doc_len:,} karakter), {len(out_parts)} parça örnekleniyor "
        f"(~{sum(len(p) for p in out_parts):,} karakter, %{coverage:.1f} kapsam, bütçe={max_total_chars:,})..."
                )
    return sampled


 
=======
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
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
