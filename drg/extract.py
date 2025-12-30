# drg/extract.py
"""
Declarative knowledge graph extraction using DSPy.
Schema'dan dinamik olarak DSPy signature'ları oluşturur - tamamen declarative.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Any, Dict
import os
import logging
import json
import dspy
from pydantic import BaseModel
from pydantic.config import ConfigDict
from unittest.mock import Mock
import re
import math

from .utils.llm_throttle import throttle_llm_calls

from .schema import (
    DRGSchema, 
    EnhancedDRGSchema, 
    Entity, 
    Relation,
    EntityType,
    RelationGroup
)
from .graph.kg_core import KGEdge

# Lazy imports for optional modules
try:
    from .optimizer import DRGOptimizer, OptimizerConfig
except ImportError:
    DRGOptimizer = None
    OptimizerConfig = None

try:
    from .coreference_resolution import resolve_coreferences
except ImportError:
    resolve_coreferences = None

try:
    from .entity_resolution import resolve_entities_and_relations
except ImportError:
    resolve_entities_and_relations = None

logger = logging.getLogger(__name__)


# Lightweight result container to avoid hard dependency on dspy.Prediction in callers/tests.
@dataclass(frozen=True)
class ExtractionResult:
    """Extraction result (prediction-like) container.
    
    Attributes:
        entities: List of (entity_name, entity_type) tuples.
        relations: List of (source, relation, target) triples.
        enriched_relations: Optional per-relation metadata aligned with `relations`.
    """
    entities: List[Tuple[str, str]]
    relations: List[Tuple[str, str, str]]
    enriched_relations: Optional[List[Dict[str, Any]]] = None


def _should_return_dspy_prediction() -> bool:
    """Decide whether it's safe/meaningful to return a real dspy.Prediction.
    
    In unit tests, `dspy` (or `dspy.Prediction`) is often patched/mocked. Returning a mocked
    Prediction breaks attribute semantics. In real runs (and optimizer runs), returning an actual
    dspy.Prediction keeps DSPy internals/optimizers happier.
    """
    pred_cls = getattr(dspy, "Prediction", None)
    if pred_cls is None:
        return False
    if isinstance(pred_cls, Mock):
        return False
    if not isinstance(pred_cls, type):
        return False
    return getattr(pred_cls, "__module__", "").startswith("dspy")


# Pydantic models for structured output (DSPy TypedPredictor compatibility)
class EntityList(BaseModel):
    """Structured output model for entity extraction."""
    model_config = ConfigDict(extra="ignore")
    entities: List[Tuple[str, str]]  # List of (entity_name, entity_type) tuples


class RelationList(BaseModel):
    """Structured output model for relation extraction."""
    model_config = ConfigDict(extra="ignore")
    relations: List[Tuple[str, str, str]]  # List of (source, relation, target) tuples


class SchemaOutput(BaseModel):
    """Structured output model for schema generation."""
    model_config = ConfigDict(extra="ignore")
    generated_schema: str  # JSON string of EnhancedDRGSchema


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


def _infer_reverse_relation_name(relation_name: str) -> Optional[str]:
    """Infer reverse relation name from relation name (domain-agnostic).
    
    Generic reverse relation detection for relations not in the pattern list.
    Works for any domain by detecting common reverse patterns.
    
    Args:
        relation_name: Relation name to infer reverse for
    
    Returns:
        Inferred reverse relation name or None if cannot infer
    """
    relation_lower = relation_name.lower()
    
    # Pattern 1: "_by" suffix → remove it
    # "produced_by" → "produces"
    if relation_lower.endswith("_by"):
        base = relation_lower[:-3]
        # Common verb forms
        if base.endswith("ed"):
            return base  # "created" → "creates" (but pattern already handles this)
        return base + "s" if not base.endswith("s") else base
    
    # Pattern 2: "_of" suffix → replace with "has_"
    # "member_of" → "has_member"
    if relation_lower.endswith("_of"):
        base = relation_lower[:-3]
        return f"has_{base}"
    
    # Pattern 3: "_from" suffix → remove it and try action verb
    if relation_lower.endswith("_from"):
        base = relation_lower[:-5]
        return base + "s" if not base.endswith("s") else base
    
    # Pattern 4: Direct action verbs → add "_by"
    # "produces" → "produced_by"
    if not relation_lower.endswith(("_by", "_of", "_from", "_in", "_at")):
        # Simple verbs - add "_by" for passive form
        if relation_lower.endswith("s"):
            return relation_lower[:-1] + "ed_by"  # "produces" → "produced_by"
        return relation_lower + "d_by"  # "create" → "created_by"
    
    return None


def _normalize_schema(schema: Union[DRGSchema, EnhancedDRGSchema]) -> DRGSchema:
    """Convert EnhancedDRGSchema to DRGSchema for internal use."""
    if isinstance(schema, EnhancedDRGSchema):
        return schema.to_legacy_schema()
    return schema


def _add_reverse_relations(
    relation_groups: List[RelationGroup],
    entity_types: List[EntityType]
) -> List[RelationGroup]:
    """Automatically add reverse relations for common bidirectional relationships.
    
    This allows extraction of relations from both directions:
    - "produces" → "produced_by"
    - "owns" → "owned_by"
    - "located_in" → "contains"
    
    Args:
        relation_groups: List of relation groups
        entity_types: List of entity types
    
    Returns:
        Updated relation groups with reverse relations added
    """
    # Comprehensive bidirectional relation patterns - domain-agnostic
    # Covers common patterns across all domains (technology, business, science, medicine, etc.)
    reverse_patterns = {
        # Production/Creation patterns
        "produces": "produced_by",
        "produced_by": "produces",
        "creates": "created_by",
        "created_by": "creates",
        "created": "created_by",
        "manufactures": "manufactured_by",
        "manufactured_by": "manufactures",
        "builds": "built_by",
        "built_by": "builds",
        "makes": "made_by",
        "made_by": "makes",
        
        # Ownership patterns
        "owns": "owned_by",
        "owned_by": "owns",
        "possesses": "possessed_by",
        "possessed_by": "possesses",
        
        # Founding/Establishment patterns
        "founds": "founded_by",
        "founded_by": "founds",
        "founded": "founded_by",
        "establishes": "established_by",
        "established_by": "establishes",
        
        # Design/Development patterns
        "designs": "designed_by",
        "designed_by": "designs",
        "designed": "designed_by",
        "develops": "developed_by",
        "developed_by": "develops",
        "programs": "programmed_by",
        "programmed_by": "programs",
        
        # Location patterns
        "located_in": "contains",
        "contains": "located_in",
        "located_at": "hosts",
        "hosts": "located_at",
        "situated_in": "contains",
        
        # Employment/Work patterns
        "works_at": "employs",
        "employs": "works_at",
        "works_for": "employs",
        "employed_by": "employs",
        
        # Membership patterns
        "member_of": "has_member",
        "has_member": "member_of",
        "part_of": "has_part",
        "has_part": "part_of",
        "belongs_to": "has_member",
        
        # Hierarchical patterns
        "parent_of": "child_of",
        "child_of": "parent_of",
        "manager_of": "reports_to",
        "reports_to": "manager_of",
        "supervisor_of": "reports_to",
        "subordinate_of": "supervises",
        
        # Relationship patterns
        "related_to": "related_to",  # Symmetric
        "connected_to": "connected_to",  # Symmetric
        "partners_with": "partners_with",  # Symmetric
        "collaborates_with": "collaborates_with",  # Symmetric
        
        # Action patterns (generic)
        "operates": "operated_by",
        "operated_by": "operates",
        "manages": "managed_by",
        "managed_by": "manages",
        "controls": "controlled_by",
        "controlled_by": "controls",
    }
    
    entity_names = {et.name for et in entity_types}
    new_relation_groups = []
    
    for rg in relation_groups:
        new_relations = list(rg.relations)
        added_reverse_relations = set()
        
        for rel in rg.relations:
            # Check if reverse relation already exists
            reverse_name = reverse_patterns.get(rel.name)
            if reverse_name and reverse_name not in added_reverse_relations:
                # Check if reverse relation doesn't already exist
                exists = any(
                    r.name == reverse_name and r.src == rel.dst and r.dst == rel.src
                    for r in new_relations
                )
                
                if not exists and rel.dst in entity_names and rel.src in entity_names:
                    # Add reverse relation
                    reverse_relation = Relation(
                        name=reverse_name,
                        src=rel.dst,  # Reverse direction
                        dst=rel.src,
                        description=f"Reverse of {rel.name}: {rel.description}",
                        detail=f"Reverse relationship: {rel.detail}"
                    )
                    new_relations.append(reverse_relation)
                    added_reverse_relations.add(reverse_name)
                    logger.debug(f"Added reverse relation: {reverse_name} ({rel.dst} -> {rel.src})")
        
        new_relation_groups.append(RelationGroup(
            name=rg.name,
            description=rg.description,
            relations=new_relations,
            examples=rg.examples
        ))
    
    return new_relation_groups


def _create_entity_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak EntityExtraction signature'ı oluştur.
    
    DSPy best practice: Minimal signature with InputField/OutputField only.
    Prompt engineering'den kaçın; schema kısıtlarını kısa ve açık tut.
    """
    normalized = _normalize_schema(schema)
    
    if isinstance(schema, EnhancedDRGSchema):
        entity_types_list = [et.name for et in schema.entity_types]
    else:
        entity_types_list = [e.name for e in normalized.entities]
    entity_types = ", ".join(entity_types_list)
    
    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema."""
        text: str = dspy.InputField(desc="Input text")
        entities: List[Tuple[str, str]] = dspy.OutputField(
            desc=f"Return entities as [(entity_name, entity_type), ...]. entity_type must be one of: {entity_types}."
        )
    
    # Store type list for debugging/validation (not for prompt shaping).
    EntityExtraction._entity_types = entity_types_list
    
    return EntityExtraction


def _create_relation_signature(schema: Union[DRGSchema, EnhancedDRGSchema]) -> type:
    """Schema'dan dinamik olarak RelationExtraction signature'ı oluştur.
    
    DSPy best practice: Minimal signature with InputField/OutputField only.
    Prompt engineering'den kaçın; schema kısıtlarını kısa ve açık tut.
    """
    normalized = _normalize_schema(schema)
    
    # Keep allowed relations concise: (name, src, dst) only.
    if isinstance(schema, EnhancedDRGSchema):
        schema_info = "\n".join([f"{r.name}: {r.src} -> {r.dst}" for rg in schema.relation_groups for r in rg.relations])
    else:
        schema_info = "\n".join([f"{r.name}: {r.src} -> {r.dst}" for r in normalized.relations])
    
    class RelationExtraction(dspy.Signature):
        """Extract relationships between provided entities under the schema."""
        text: str = dspy.InputField(desc="Input text (current chunk)")
        entities: List[Tuple[str, str]] = dspy.InputField(
            desc="Entities as [(name, type), ...]."
        )
        relations: List[Tuple[str, str, str]] = dspy.OutputField(
            desc=f"Return relations as [(source, relation, target), ...]. Must be valid under schema:\n{schema_info}"
        )
        # NOTE: Deliberately minimal signature: we do NOT ask the model for extra metadata fields.
        # Temporal / negation / confidence can be produced via deterministic post-processing if needed.
    
    # Store relation info for debugging/validation (not for prompt shaping).
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
    
    def forward(self, text: str, context_entities: Optional[List[Tuple[str, str]]] = None) -> ExtractionResult:
        """Extract entities and relations using DSPy TypedPredictor.
        
        Args:
            text: Input text to extract from
            context_entities: Optional list of entities from previous chunks (for cross-chunk relationships)
        
        Returns:
            dspy.Prediction with entities and relations as lists of tuples
        
        Note:
            - DSPy TypedPredictor automatically ensures correct output type (EntityList/RelationList)
            - DSPy will retry until output matches the expected Pydantic model structure
            - No manual validation needed - DSPy handles type checking automatically
            - context_entities allows linking relationships across chunks
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
            entities_raw = getattr(entity_result, 'entities', '[]')
            
            # Handle both string and list outputs
            if isinstance(entities_raw, list):
                # LLM already returned a list
                entities_list = []
                for item in entities_raw:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        entities_list.append((str(item[0]), str(item[1])))
            elif isinstance(entities_raw, str):
                # LLM returned a JSON string, parse it
                try:
                    entities = _parse_json_output(entities_raw, expected_format="array")
                    entities_list = []
                    for item in entities:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            entities_list.append((str(item[0]), str(item[1])))
                except ValueError as e:
                    logger.error(f"Entity extraction JSON parsing failed: {e}")
                    raise RuntimeError(f"Failed to parse entity extraction output: {e}") from e
            else:
                logger.error(f"Entity extraction returned unexpected type: {type(entities_raw)}")
                raise RuntimeError(f"Entity extraction returned invalid type: {type(entities_raw)}")
        
        # Merge with context entities (for cross-chunk relationship discovery)
        if context_entities:
            # Combine entities, avoiding duplicates
            existing_entity_names = {(name.lower(), etype) for name, etype in entities_list}
            for name, etype in context_entities:
                if (name.lower(), etype) not in existing_entity_names:
                    entities_list.append((name, etype))
            logger.info(f"Merged {len(context_entities)} context entities, total: {len(entities_list)} entities")
        
        # Filter empty names (data cleaning only)
        if entities_list:
            entities_list = [(name, etype) for name, etype in entities_list if name.strip()]
        
        logger.info(f"Entity extraction tamamlandı: {len(entities_list)} entity bulundu")
        
        # Step 2: Extract relations (entities'i input olarak ver, context entities dahil)
        # CRITICAL: entities_list now contains current chunk entities + all context entities
        # This enables cross-chunk relationship discovery (e.g., chunk 15 can relate to entities from chunk 1)
        context_count = len(context_entities) if context_entities else 0
        current_count = len(entities_list) - context_count
        logger.info(
            f"Relation extraction başlatılıyor: {current_count} current entities + {context_count} context entities = {len(entities_list)} total entities. "
            f"LLM can extract relationships between ANY of these entities, enabling cross-chunk relationship discovery."
        )
        
        if self._use_typed_predictor:
            # TypedPredictor expects List[Tuple] directly, not JSON string
            # Reference: https://github.com/stanfordnlp/dspy
            relation_result = self.relation_extractor(
                text=text,
                entities=entities_list
            )
            # relation_result is RelationList, access fields
            if isinstance(relation_result, RelationList):
                relations_list = relation_result.relations  # List[Tuple[str, str, str]]
            else:
                # Fallback: Try to get relations field (should not happen if TypedPredictor works correctly)
                relations_list = getattr(relation_result, "relations", [])
                logger.warning(f"Expected RelationList, got {type(relation_result).__name__}")
            
            # Ensure it's a list of tuples
            if not isinstance(relations_list, list):
                error_msg = f"Expected list, got {type(relations_list).__name__}"
                logger.error(f"Relation extraction: {error_msg}")
                raise RuntimeError(f"Relation extraction returned invalid type: {error_msg}")
            # Minimal signature: do NOT request extra metadata fields from the LLM.
            # Compute deterministic heuristics (English-first, conservative) for negation/temporal.
            heur = _infer_relation_metadata_heuristic(text=text, relations=relations_list)
            confidence_scores = None
            temporal_info = heur.get("temporal_info")
            negations = heur.get("negations")
        else:
            # Fallback: Convert entities to JSON string (old method - TypedPredictor not available)
            entities_json = json.dumps(entities_list)
            relation_result = self.relation_extractor(
                text=text,
                entities=entities_json
            )
            relations_raw = getattr(relation_result, 'relations', '[]')
            
            # Handle both string and list outputs
            if isinstance(relations_raw, list):
                # LLM already returned a list
                relations_list = []
                for item in relations_raw:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        relations_list.append((str(item[0]), str(item[1]), str(item[2])))
            elif isinstance(relations_raw, str):
                # LLM returned a JSON string, parse it
                try:
                    relations = _parse_json_output(relations_raw, expected_format="array")
                    relations_list = []
                    for item in relations:
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            relations_list.append((str(item[0]), str(item[1]), str(item[2])))
                except ValueError as e:
                    logger.error(f"Relation extraction JSON parsing failed: {e}")
                    raise RuntimeError(f"Failed to parse relation extraction output: {e}") from e
            else:
                logger.error(f"Relation extraction returned unexpected type: {type(relations_raw)}")
                raise RuntimeError(f"Relation extraction returned invalid type: {type(relations_raw)}")
            # Minimal signature: compute deterministic heuristics for negation/temporal.
            heur = _infer_relation_metadata_heuristic(text=text, relations=relations_list)
            confidence_scores = None
            temporal_info = heur.get("temporal_info")
            negations = heur.get("negations")
        
        # DSPy TypedPredictor automatically ensures correct output format
        # No manual validation needed - DSPy handles type checking and retries automatically
        
        logger.info(f"Relation extraction tamamlandı: {len(relations_list)} relation bulundu")
        
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


def extract_from_chunks(
    chunks: List[Dict[str, Any]],
    schema: Union[DRGSchema, EnhancedDRGSchema],
    enable_cross_chunk_relationships: bool = True,
    enable_entity_resolution: bool = True,
    enable_coreference_resolution: bool = False,
    enable_implicit_relationships: bool = True,
    enable_cross_chunk_context_snippets: bool = True,
    max_cross_chunk_context_chunks: int = 3,
    cross_chunk_snippet_chars: int = 350,
    max_cross_chunk_context_chars: int = 1200,
    min_anchor_entity_len: int = 3,
    max_anchor_entities: int = 8,
        two_pass_extraction: bool = True,  # Default True for better cross-chunk relationship discovery
    embedding_provider=None,  # Optional embedding provider for entity/coreference resolution
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Extract entities and relations from multiple chunks with cross-chunk relationship support.
    
    **This is the recommended function for long texts** that need to be chunked. Unlike `extract_typed()`,
    this function maintains entity context across chunks, allowing discovery of relationships between
    entities that appear in different chunks (e.g., entity A in chunk 1, entity B in chunk 15).
    
    **Two extraction modes:**
    
    1. **Single-pass mode** (two_pass_extraction=False): Processes chunks sequentially, maintaining incremental context
       - Processes chunks one by one
       - Maintains a context_entities list that accumulates entities from all previous chunks
       - For each chunk, passes context_entities to relation extraction
       - More efficient but may miss some cross-chunk relationships between distant chunks
    
    2. **Two-pass mode** (two_pass_extraction=True, default): Optimal for capturing all cross-chunk relationships
       - **Pass 1**: Extract ALL entities from ALL chunks (entity extraction only, no relations)
       - **Pass 2**: Re-process ALL chunks with GLOBAL entity context (all entities from pass 1)
       - LLM can see all entities when extracting relations, enabling comprehensive cross-chunk relationship discovery
       - Ensures no relationships are missed due to entity visibility limitations
       - More accurate but requires processing chunks twice (recommended for best results)
    
    **Default behavior**: two_pass_extraction=True is now the default for optimal cross-chunk relationship discovery.
    This ensures that relationships between entities in distant chunks are captured, regardless of document domain or content type.
    
    Example:
        from drg.chunking import create_chunker
        from drg.extract import extract_from_chunks
        
        # Chunk long text
        chunker = create_chunker(strategy="token_based", chunk_size=768, overlap_ratio=0.15)
        chunks = chunker.chunk(text)
        
        # Extract with cross-chunk relationship discovery
        entities, relations = extract_from_chunks(
            chunks=[{"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks],
            schema=schema,
            enable_cross_chunk_relationships=True,  # Enable cross-chunk relationships
            enable_entity_resolution=True,
            enable_coreference_resolution=True
        )
    
    Args:
        chunks: List of chunk dictionaries with 'text' field (required) and optional 'chunk_id', 'metadata'
        schema: DRGSchema defining allowed entity types and relations
        enable_cross_chunk_relationships: Whether to maintain entity context across chunks (default: True)
                                        **IMPORTANT**: Set to False only if you don't need cross-chunk relationships.
                                        Setting to False will lose relationships between entities in different chunks.
        enable_entity_resolution: Whether to enable entity resolution (merges duplicate entity names)
        enable_coreference_resolution: Whether to enable coreference resolution (resolves pronouns/references)
        embedding_provider: Optional embedding provider for semantic similarity-based entity resolution 
                           and coreference disambiguation. If provided, improves pronoun resolution in 
                           multi-entity contexts (e.g., "EntityA and EntityB met. He spoke about Topic." 
                           → semantic similarity helps disambiguate "He").
        two_pass_extraction: If True (default), uses two-pass extraction for better cross-chunk relationships:
                            Pass 1: Extract all entities from all chunks
                            Pass 2: Extract relations with global entity context (all entities visible)
                            If False, uses single-pass mode (incremental context, faster but may miss some relationships)
                            **RECOMMENDED: Keep True for best accuracy, especially for long documents**
    
    Returns:
        Tuple of (entities_typed, triples) where:
        - entities_typed: List of (entity_name, entity_type) tuples
        - triples: List of (source, relation, target) tuples
    """
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

    # Get base extractor
    extractor = _get_extractor(schema)

    # If DSPy LM is not configured, avoid raising and behave like "mock mode" by default *only*
    # when using the real KGExtractor (unit tests may patch _get_extractor with a mock extractor).
    lm = getattr(getattr(dspy, "settings", None), "lm", None)
    if lm is None and isinstance(extractor, KGExtractor):
        if os.getenv("DRG_REQUIRE_LM", "").lower() in {"1", "true", "yes"}:
            raise ValueError(
                "No DSPy LM is loaded. Configure LM via environment variables (e.g., DRG_MODEL + API key) "
                "or unset DRG_REQUIRE_LM to allow mock-mode empty extraction."
            )
        logger.warning("No DSPy LM configured; returning empty extraction (mock mode).")
        if return_enriched:
            return [], [], []
        return [], []
    
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
    
    # Process enriched_relations if available (contains temporal, confidence, negation info)
    enriched_relations = None
    enriched_raw = getattr(result, "enriched_relations", None)
    if isinstance(enriched_raw, list) and enriched_raw:
        enriched_relations = enriched_raw
        
        # Filter out negated relations and low-confidence relations
        filtered_triples = []
        filtered_enriched = []
        for i, rel_dict in enumerate(enriched_relations):
            # Skip negated relations
            if rel_dict.get('is_negated', False):
                logger.debug(f"Filtered out negated relation: {rel_dict.get('relation')}")
                continue
            
            # Skip low-confidence relations if min_confidence is set
            if min_confidence is not None:
                confidence = rel_dict.get('confidence')
                if confidence is None:
                    # If no confidence provided, keep it (don't filter)
                    logger.debug(f"No confidence score for relation {rel_dict.get('relation')}, keeping")
                elif confidence < min_confidence:
                    logger.debug(
                        f"Filtered out low-confidence relation {rel_dict.get('relation')} "
                        f"(confidence: {confidence:.2f} < {min_confidence:.2f})"
                    )
                    continue
            
            # Keep this relation
            filtered_triples.append(triples[i] if i < len(triples) else rel_dict['relation'])
            filtered_enriched.append(rel_dict)
        
        triples = filtered_triples
        enriched_relations = filtered_enriched  # Keep enriched metadata aligned with filtered triples
        
        # Log filtering results
        if min_confidence is not None:
            filtered_count = len(result.enriched_relations) - len(filtered_enriched)
            if filtered_count > 0:
                logger.info(
                    f"Confidence filtering: {filtered_count} relationships filtered out "
                    f"(confidence < {min_confidence:.2f}), {len(filtered_enriched)} remaining"
                )
    
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
    # Reverse relations desteği: "produced_by" gibi reverse relation'ları da kabul et
    if isinstance(schema, EnhancedDRGSchema):
        valid_triples = []
        # Use the same comprehensive reverse relation patterns as _add_reverse_relations
        # This ensures consistency across the codebase
        reverse_patterns = {
            # Production/Creation patterns
            "produces": "produced_by", "produced_by": "produces",
            "creates": "created_by", "created_by": "creates",
            "created": "created_by",
            "manufactures": "manufactured_by", "manufactured_by": "manufactures",
            "builds": "built_by", "built_by": "builds",
            "makes": "made_by", "made_by": "makes",
            # Ownership patterns
            "owns": "owned_by", "owned_by": "owns",
            "possesses": "possessed_by", "possessed_by": "possesses",
            # Founding patterns
            "founds": "founded_by", "founded_by": "founds", "founded": "founded_by",
            "establishes": "established_by", "established_by": "establishes",
            # Design/Development patterns
            "designs": "designed_by", "designed_by": "designs", "designed": "designed_by",
            "develops": "developed_by", "developed_by": "develops",
            "programs": "programmed_by", "programmed_by": "programs",
            # Location patterns
            "located_in": "contains", "contains": "located_in",
            "located_at": "hosts", "hosts": "located_at", "situated_in": "contains",
            # Employment patterns
            "works_at": "employs", "employs": "works_at",
            "works_for": "employs", "employed_by": "employs",
            # Membership patterns
            "member_of": "has_member", "has_member": "member_of",
            "part_of": "has_part", "has_part": "part_of", "belongs_to": "has_member",
            # Hierarchical patterns
            "parent_of": "child_of", "child_of": "parent_of",
            "manager_of": "reports_to", "reports_to": "manager_of",
            "supervisor_of": "reports_to", "subordinate_of": "supervises",
            # Action patterns
            "operates": "operated_by", "operated_by": "operates",
            "manages": "managed_by", "managed_by": "manages",
            "controls": "controlled_by", "controlled_by": "controls",
        }
        reverse_patterns_inv = {v: k for k, v in reverse_patterns.items() if k != v}
        
        for s, r, o in triples:
            s_type = next((etype for name, etype in valid_entities if name == s), None)
            o_type = next((etype for name, etype in valid_entities if name == o), None)
            
            if not (s_type and o_type):
                continue
            
            # Check direct relation first
            if schema.is_valid_relation(r, s_type, o_type):
                valid_triples.append((s, r, o))
                continue
            
            # Check reverse relation (e.g., "produced_by" instead of "produces")
            # Strategy 1: Pattern-based reverse relation detection
            if r in reverse_patterns_inv:
                reverse_rel = reverse_patterns_inv[r]
                # Try: reverse relation in correct direction (o_type -> s_type)
                if schema.is_valid_relation(reverse_rel, o_type, s_type):
                    # Convert: (source, reverse_rel, target) → (target, direct_rel, source)
                    # Example: (iPhone, produced_by, Apple) → (Apple, produces, iPhone)
                    valid_triples.append((o, reverse_rel, s))
                    logger.debug(f"Converted reverse relation (pattern-based): ({s}, {r}, {o}) -> ({o}, {reverse_rel}, {s})")
                    continue
                # Also try: direct relation in reverse direction (if reverse_rel is also in patterns)
                elif reverse_rel in reverse_patterns:
                    direct_rel = reverse_patterns[reverse_rel]
                    if schema.is_valid_relation(direct_rel, o_type, s_type):
                        # Convert: (source, reverse_rel, target) → (target, direct_rel, source)
                        valid_triples.append((o, direct_rel, s))
                        logger.debug(f"Converted reverse relation (pattern-based): ({s}, {r}, {o}) -> ({o}, {direct_rel}, {s})")
                        continue
            
            # Strategy 2: Generic reverse relation detection (domain-agnostic)
            generic_reverse_rel = _infer_reverse_relation_name(r)
            if generic_reverse_rel and generic_reverse_rel != r:
                # Try reverse direction: swap source and target
                if schema.is_valid_relation(generic_reverse_rel, o_type, s_type):
                    valid_triples.append((o, generic_reverse_rel, s))
                    logger.debug(f"Converted reverse relation (generic): ({s}, {r}, {o}) -> ({o}, {generic_reverse_rel}, {s})")
                    continue
                # Also try: if current relation might be reverse, try removing suffix
                if r.endswith("_by") or r.endswith("_of") or r.endswith("_from"):
                    base_name = r.rsplit("_", 1)[0] if "_" in r else r
                    if schema.is_valid_relation(base_name, o_type, s_type):
                        valid_triples.append((o, base_name, s))
                        logger.debug(f"Converted reverse relation (suffix removal): ({s}, {r}, {o}) -> ({o}, {base_name}, {s})")
                        continue
    else:
        # Legacy DRGSchema support
        rel_types = {(r.src, r.name, r.dst) for r in normalized.relations}
        # Use the same comprehensive reverse relation patterns (consistent with EnhancedDRGSchema)
        reverse_patterns = {
            # Production/Creation patterns
            "produces": "produced_by", "produced_by": "produces",
            "creates": "created_by", "created_by": "creates",
            "created": "created_by",
            "manufactures": "manufactured_by", "manufactured_by": "manufactures",
            "builds": "built_by", "built_by": "builds",
            "makes": "made_by", "made_by": "makes",
            # Ownership patterns
            "owns": "owned_by", "owned_by": "owns",
            "possesses": "possessed_by", "possessed_by": "possesses",
            # Founding patterns
            "founds": "founded_by", "founded_by": "founds", "founded": "founded_by",
            "establishes": "established_by", "established_by": "establishes",
            # Design/Development patterns
            "designs": "designed_by", "designed_by": "designs", "designed": "designed_by",
            "develops": "developed_by", "developed_by": "develops",
            "programs": "programmed_by", "programmed_by": "programs",
            # Location patterns
            "located_in": "contains", "contains": "located_in",
            "located_at": "hosts", "hosts": "located_at", "situated_in": "contains",
            # Employment patterns
            "works_at": "employs", "employs": "works_at",
            "works_for": "employs", "employed_by": "employs",
            # Membership patterns
            "member_of": "has_member", "has_member": "member_of",
            "part_of": "has_part", "has_part": "part_of", "belongs_to": "has_member",
            # Hierarchical patterns
            "parent_of": "child_of", "child_of": "parent_of",
            "manager_of": "reports_to", "reports_to": "manager_of",
            "supervisor_of": "reports_to", "subordinate_of": "supervises",
            # Action patterns
            "operates": "operated_by", "operated_by": "operates",
            "manages": "managed_by", "managed_by": "manages",
            "controls": "controlled_by", "controlled_by": "controls",
        }
        reverse_patterns_inv = {v: k for k, v in reverse_patterns.items() if k != v}
        
        valid_triples = []
        for s, r, o in triples:
            s_type = next((etype for name, etype in valid_entities if name == s), None)
            o_type = next((etype for name, etype in valid_entities if name == o), None)
            
            if not (s_type and o_type):
                continue
            
            # Check direct relation first
            if (s_type, r, o_type) in rel_types:
                valid_triples.append((s, r, o))
                continue
            
            # Check reverse relation - try both conversion directions
            # Strategy 1: Pattern-based reverse relation detection
            if r in reverse_patterns_inv:
                reverse_rel = reverse_patterns_inv[r]
                # Try: reverse relation in correct direction (o_type -> s_type)
                if (o_type, reverse_rel, s_type) in rel_types:
                    # Convert: (source, reverse_rel, target) → (target, reverse_rel, source)
                    valid_triples.append((o, reverse_rel, s))
                    logger.debug(f"Converted reverse relation (pattern-based): ({s}, {r}, {o}) -> ({o}, {reverse_rel}, {s})")
                    continue
                # Also try: direct relation if reverse_rel has a pattern
                elif reverse_rel in reverse_patterns:
                    direct_rel = reverse_patterns[reverse_rel]
                    if (o_type, direct_rel, s_type) in rel_types:
                        valid_triples.append((o, direct_rel, s))
                        logger.debug(f"Converted reverse relation (pattern-based): ({s}, {r}, {o}) -> ({o}, {direct_rel}, {s})")
                        continue
            
            # Strategy 2: Generic reverse relation detection (domain-agnostic)
            generic_reverse_rel = _infer_reverse_relation_name(r)
            if generic_reverse_rel and generic_reverse_rel != r:
                # Try reverse direction: swap source and target
                if (o_type, generic_reverse_rel, s_type) in rel_types:
                    valid_triples.append((o, generic_reverse_rel, s))
                    logger.debug(f"Converted reverse relation (generic): ({s}, {r}, {o}) -> ({o}, {generic_reverse_rel}, {s})")
                    continue
                # Also try: if current relation might be reverse, try removing suffix
                if r.endswith("_by") or r.endswith("_of") or r.endswith("_from"):
                    base_name = r.rsplit("_", 1)[0] if "_" in r else r
                    if (o_type, base_name, s_type) in rel_types:
                        valid_triples.append((o, base_name, s))
                        logger.debug(f"Converted reverse relation (suffix removal): ({s}, {r}, {o}) -> ({o}, {base_name}, {s})")
                        continue
    
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
                    use_nlp=True,  # Use NLP if available, falls back to heuristics otherwise
                    embedding_provider=embedding_provider,
                    language=os.getenv("DRG_LANGUAGE", "en"),
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
                    similarity_threshold=0.65,  # Lowered from 0.85 for better recall
                    adaptive_threshold=True,  # Adaptive threshold for short names
                    embedding_provider=embedding_provider,
                    use_embedding=bool(embedding_provider),
                )
                logger.info("Entity resolution applied successfully")
            except Exception as e:
                logger.warning(f"Entity resolution failed: {e}, continuing without resolution")

    # Optional: deterministic implicit relationship inference (schema-gated).
    if enable_implicit_relationships and valid_entities and text:
        inferred = _infer_implicit_relations(
            text=text, entities=valid_entities, schema=schema, existing_triples=valid_triples
        )
        if inferred:
            existing = set(valid_triples)
            for t in inferred:
                if t not in existing:
                    valid_triples.append(t)
                    existing.add(t)
    
    # Map enriched_relations to valid triples before coreference/entity resolution
    # (After resolution, entity names may change, so we map before resolution)
    valid_enriched = []
    if return_enriched and enriched_relations:
        # Create mapping from triple to enriched metadata
        triple_to_enriched = {}
        for rel_dict in enriched_relations:
            rel_tuple = rel_dict.get('relation')
            if rel_tuple:
                triple_to_enriched[rel_tuple] = rel_dict
        
        # Map valid triples to enriched metadata (after schema validation)
        valid_triple_set = set(valid_triples)
        for triple in valid_triples:
            if triple in triple_to_enriched:
                valid_enriched.append(triple_to_enriched[triple])
            else:
                # If enriched metadata not available, create minimal entry
                valid_enriched.append({
                    "relation": triple,
                    "confidence": None,
                    "temporal": None,
                    "is_negated": False,
                })
    
    # Coreference and entity resolution may change entity names in triples
    # If enriched_relations is needed, we need to update the mapping after resolution
    # For now, enriched_relations are mapped to pre-resolution triples
    # Note: After coreference/entity resolution, enriched_relations may be slightly misaligned
    # This is acceptable since temporal/confidence info is typically entity-agnostic
    
    # Return enriched relations if requested
    if return_enriched:
        return valid_entities, valid_triples, valid_enriched
    
    return valid_entities, valid_triples


def _infer_implicit_relations(
    text: str,
    entities: List[Tuple[str, str]],
    schema: Union[DRGSchema, EnhancedDRGSchema],
    existing_triples: Optional[List[Tuple[str, str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    """Infer a small set of implicit relations from surface patterns (schema-gated).
    
    Only emits relations that exist in the provided schema. This complements LLM extraction
    for cases like "Tesla's Gigafactory" / "Tesla'nın Gigafactory'si".
    """
    if not text or not entities:
        return []
    
    normalized = _normalize_schema(schema)
    # DRGSchema doesn't expose is_valid_relation, so we build a set.
    legacy_rel_types = {(r.src, r.name, r.dst) for r in normalized.relations}
    
    def _allows(rel: str, s_type: str, o_type: str) -> bool:
        if isinstance(schema, EnhancedDRGSchema):
            return schema.is_valid_relation(rel, s_type, o_type)
        return (s_type, rel, o_type) in legacy_rel_types
    
    type_map: Dict[str, str] = {name: etype for name, etype in entities if name and etype}
    entity_names = [name for name, _ in entities if name]
    if len(entity_names) < 2:
        return []
    
    text_l = text.lower()
    entity_names_sorted = sorted(entity_names, key=lambda s: len(s), reverse=True)
    
    # Candidate implicit relations in priority order (conservative).
    # NOTE: We intentionally avoid overly generic "has" and reverse-like "part_of" here
    # because possessives are ambiguous; schema authors can still express composition via has_part.
    candidate_rels = ["owns", "has_part"]
    
    inferred: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set(existing_triples or [])
    
    def _try_add(a: str, b: str) -> None:
        a_type = type_map.get(a)
        b_type = type_map.get(b)
        if not a_type or not b_type:
            return
        for rel in candidate_rels:
            if _allows(rel, a_type, b_type):
                t = (a, rel, b)
                if t not in seen:
                    inferred.append(t)
                    seen.add(t)
                return
    
    # Possessive detection (English + Turkish) with word-boundary regex for safety.
    # We keep this conservative to avoid substring collisions.
    gen_suffixes = ["nın", "nin", "nun", "nün", "ın", "in", "un", "ün"]

    def _has_possessive(a: str, b: str) -> bool:
        a_esc = re.escape(a)
        b_esc = re.escape(b)
        # English: "A's B" (apostrophe variants)
        en_pat = rf"(?i)(?<!\w){a_esc}(?:'s|’s)\s+{b_esc}(?!\w)"
        if re.search(en_pat, text):
            return True
        # Turkish: "A'nın B" with optional apostrophe and common genitive suffixes
        # Example: Tesla'nın Gigafactory'si, Tesla nin Gigafactory
        suf_alt = "|".join(re.escape(s) for s in gen_suffixes)
        tr_pat = rf"(?i)(?<!\w){a_esc}(?:'?\s*(?:{suf_alt}))\s+{b_esc}(?!\w)"
        return re.search(tr_pat, text) is not None
    for a in entity_names_sorted:
        for b in entity_names_sorted:
            if a == b:
                continue
            if _has_possessive(a, b):
                _try_add(a, b)
    
    # Two-hop inference (schema-gated, input-agnostic):
    # If (A owns/has_part/has B) and (B located_in/located_at ... L) then infer (A operates_in/located_in ... L)
    # when such relations exist in schema.
    if existing_triples:
        # Keep conservative: only treat explicit-ish possession/composition as ownership edge.
        ownership_rels = {"owns", "has_part"}
        location_rels = {"located_in", "located_at", "hosts", "contains"}
        candidate_owner_location_rels = ["operates_in", "based_in", "headquartered_in"]

        # Optional evidence cue: if the text contains a generic "operation" signal, allow operates_in-like inference.
        # This is intentionally minimal and conservative; if not present we simply don't infer.
        operation_cue_patterns = [
            r"(?i)\boperat(?:e|es|ing|ed)\b",
            r"(?i)\brun(?:s|ning|ned)?\b",
            r"(?i)\bmanage(?:s|d|ment|ing)?\b",
            r"(?i)\bemploy(?:s|ed|ing)?\b",
            r"(?i)\bwork(?:s|ed|ing)?\b",
            r"(?i)\bfaaliyet\b",
            r"(?i)\bçalıştır(?:ıyor|di|mak|ma)?\b",
            r"(?i)\bçalış(?:ıyor|tı|mak|ma)?\b",
        ]
        has_operation_cue = any(re.search(p, text) for p in operation_cue_patterns)
        if not has_operation_cue:
            return inferred

        # Build quick maps
        owner_to_asset: Dict[str, Set[str]] = {}
        asset_to_location: Dict[str, Set[str]] = {}

        combined = list(existing_triples) + inferred
        for s, r, o in combined:
            if r in ownership_rels:
                owner_to_asset.setdefault(s, set()).add(o)
            if r in location_rels:
                asset_to_location.setdefault(s, set()).add(o)

        for owner, assets in owner_to_asset.items():
            owner_type = type_map.get(owner)
            if not owner_type:
                continue
            for asset in assets:
                # Extra safety: require the asset mention itself in text to avoid
                # accidental type-shaped inference from unrelated triples.
                if asset.lower() not in text_l:
                    continue
                for loc in asset_to_location.get(asset, set()):
                    loc_type = type_map.get(loc)
                    if not loc_type:
                        continue
                    for rel in candidate_owner_location_rels:
                        if _allows(rel, owner_type, loc_type):
                            t = (owner, rel, loc)
                            if t not in seen:
                                inferred.append(t)
                                seen.add(t)
                            break

    return inferred


def create_kgedge_from_triple(
    triple: Tuple[str, str, str],
    enriched_metadata: Optional[Dict[str, Any]] = None,
    relationship_detail: Optional[str] = None,
) -> "KGEdge":
    """Create a KGEdge from a triple tuple with optional enriched metadata (temporal, confidence, negation).
    
    This helper function makes it easy to create KGEdge objects with temporal information
    from extracted triples and enriched_relations metadata.
    
    Args:
        triple: (source, relation, target) tuple
        enriched_metadata: Optional dict from enriched_relations with keys:
            - "temporal": {"start": str, "end": str} - ISO format dates
            - "confidence": float - Confidence score (0.0-1.0)
            - "is_negated": bool - Whether relation is negated
        relationship_detail: Optional detail string (defaults to "source relation target")
    
    Returns:
        KGEdge object with temporal information, confidence, and negation flags
    
    Example:
        entities, triples, enriched = extract_typed(text, schema, return_enriched=True)
        for triple, enriched_dict in zip(triples, enriched):
            edge = create_kgedge_from_triple(triple, enriched_dict)
            kg.add_edge(edge)
    """
    from .graph.kg_core import KGEdge
    
    source, relation, target = triple
    if relationship_detail is None:
        relationship_detail = f"{source} {relation} {target}"
    
    # Extract temporal info
    start_time = None
    end_time = None
    if enriched_metadata and enriched_metadata.get('temporal'):
        temporal = enriched_metadata['temporal']
        if isinstance(temporal, dict):
            start_time = temporal.get('start')
            end_time = temporal.get('end')
    
    # Extract confidence
    confidence = enriched_metadata.get('confidence') if enriched_metadata else None
    
    # Extract negation
    is_negated = enriched_metadata.get('is_negated', False) if enriched_metadata else False
    
    return KGEdge(
        source=source,
        target=target,
        relationship_type=relation,
        relationship_detail=relationship_detail,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        is_negated=is_negated,
    )


# Backward-compatible thin wrapper
def extract_triples(text: str, schema: Union[DRGSchema, EnhancedDRGSchema]) -> List[Tuple[str, str, str]]:
    """Extract triples from text (backward compatibility)."""
    _, triples = extract_typed(text, schema)
    return triples


class SchemaGeneration(dspy.Signature):
    """Generate EnhancedDRGSchema from the given text.
    
    Output must be valid JSON matching EnhancedDRGSchema: entity_types and relation_groups.
    Derive entity types, examples/properties, relation groups and relations from the text (dataset-agnostic).
    """
    text: str = dspy.InputField(desc="Input text to analyze for schema generation")
    generated_schema: str = dspy.OutputField(
        desc="Return ONLY valid JSON for EnhancedDRGSchema with keys: "
             "'entity_types' (name, description, examples, properties) and "
             "'relation_groups' (name, description, relations[] with name, source, target, description, detail). "
             "Use entity TYPE names (e.g., Person, Company) as source/target (not entity instances). "
             "IMPORTANT formatting rules: output MUST be strict JSON (double quotes), no trailing commas, no comments, no extra text. "
             "Keep it compact to avoid truncation: max 10 entity_types; max 8 relation_groups; max 10 relations per group. "
             "For EntityType.properties, output a JSON object/dict (not a list)."
    )


def generate_schema_from_text(text: str, max_retries: int = 3, retry_delay: float = 2.0) -> EnhancedDRGSchema:
    """
    Metinden otomatik olarak enhanced şema oluştur.
    
    Bu fonksiyon, verilen metni analiz ederek uygun entity tipleri (properties ve examples ile),
    relation grupları ve detaylı açıklamalar çıkarır ve bir EnhancedDRGSchema nesnesi döndürür.
    
    **Domain-Agnostic**: Her domain (technology, business, science, medicine, history, literature, etc.)
    ve her input type (articles, reports, books, transcripts, etc.) için çalışır.
    
    **Intelligent Sampling Strategy**: Uzun dokümanlar için akıllı örnekleme stratejisi kullanır:
    - 100k karakterlik doküman → 12 parça, %45 kapsam (~45k karakter)
    - 200k karakterlik doküman → 15 parça, %45 kapsam (~90k karakter)
    - 500k+ karakterlik doküman → 20 parça, %45 kapsam (~225k karakter)
    - Eşit aralıklı örnekleme: Dokümanın her bölümünden eşit oranda örnek alınır
    - Bu sayede kritik entity tipleri ve ilişkiler örneklenmeyen kısımda kalmaz
    
    NOTE: This function uses TypedPredictor with minimal prompt for schema generation,
    following DSPy's declarative philosophy. The LLM is responsible for creating rich,
    interconnected schemas based on the text content.
    
    Args:
        text: Analiz edilecek metin (herhangi bir domain veya format)
        max_retries: Rate limit hatası durumunda maksimum deneme sayısı (DEPRECATED - retry handled by DSPy LM)
        retry_delay: Denemeler arası bekleme süresi (saniye) (DEPRECATED - retry handled by DSPy LM)
    
    Returns:
        EnhancedDRGSchema: Metne uygun detaylı şema (domain-agnostic)
    """
    # LLM'i konfigüre et
    _configure_llm_auto()
    
    sample_text = _sample_text_for_schema_generation(text)
    
    # Schema generation signature'ı oluştur
    # TypedPredictor kullanarak structured output garantisi verelim
    try:
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
        
        logger.info("Schema generation tamamlandı")
    except Exception as e:
        logger.error(f"Schema generation failed: {e}")
        raise RuntimeError(f"Schema generation failed: {e}. Check your LLM configuration and API keys.") from e
    
    # Parse JSON schema
    # DSPy TypedPredictor automatically retries until correct format is returned
    try:
        # Debug: Log raw schema output (first 500 chars)
        logger.debug(f"Raw schema output (first 500 chars): {schema_str[:500]}")
        schema_data = _parse_json_output(schema_str, expected_format="object")
        logger.debug(
            f"Parsed schema keys: {list(schema_data.keys()) if isinstance(schema_data, dict) else 'Not a dict'}"
        )
    except ValueError as e:
        logger.error(f"Failed to parse schema JSON: {e}")
        logger.error(f"Raw schema output (first 1000 chars): {schema_str[:1000]}")
        raise RuntimeError(
            f"Schema JSON parsing failed: {e}. "
            "This usually means the LLM output format is incorrect. "
            "Check your LLM configuration or try a different model."
        ) from e
    
    # Validate schema_data is not empty
    if not schema_data or (isinstance(schema_data, dict) and not schema_data):
        logger.error("Parsed schema JSON is empty")
        logger.error(f"Schema data type: {type(schema_data)}, value: {schema_data}")
        logger.error(f"Raw schema output: {schema_str[:1000]}")
        raise RuntimeError(
            "Schema generation returned empty schema. "
            "The LLM may need a better prompt or different configuration."
        )

    # Parse/validate schema strictly (no instance-vs-type heuristics).
    try:
        schema = EnhancedDRGSchema.from_dict(schema_data)
    except Exception as e:
        # Backward-compatible conversion from legacy schema shape if present.
        if isinstance(schema_data, dict) and "entities" in schema_data and "relations" in schema_data:
            entity_types = [
                EntityType(
                    name=e["name"],
                    description=e.get("description") or "Auto-generated entity type",
                    examples=e.get("examples", []) if isinstance(e.get("examples", []), list) else [],
                    properties=e.get("properties", {}) if isinstance(e.get("properties", {}), dict) else {},
                )
                for e in schema_data.get("entities", [])
                if isinstance(e, dict) and e.get("name")
            ]
            relations = [
                Relation(
                    name=r["name"],
                    src=r.get("source", r.get("src", "")),
                    dst=r.get("target", r.get("dst", "")),
                    description=r.get("description", ""),
                    detail=r.get("detail", ""),
                )
                for r in schema_data.get("relations", [])
                if isinstance(r, dict) and r.get("name")
            ]
            if not entity_types or not relations:
                raise RuntimeError(f"Legacy schema conversion failed: {e}") from e
            schema = EnhancedDRGSchema(
                entity_types=entity_types,
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


def _sample_text_for_schema_generation(text: str) -> str:
    """Deterministic, input-agnostic sampling for schema generation.
    
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


 
