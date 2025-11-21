"""
DSPy-style Optimizer for relationship refinement and iterative improvement.
"""
import json
from typing import List, Tuple, Optional, Dict, Any
import dspy
from dspy.teleprompt import BootstrapFewShot

from .schema import DRGSchema
from .extract import KGExtractor


def _parse_entities(entities) -> List[Tuple[str, str]]:
    """Parse entities from DSPy structured output (List[Tuple] veya string)."""
    if not entities:
        return []
    
    # Eğer zaten list/tuple ise direkt kullan
    if isinstance(entities, list):
        result = []
        for item in entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((str(item[0]), str(item[1])))
        return result
    
    # Eğer string ise JSON parse et (backward compatibility)
    if isinstance(entities, str):
        try:
            parsed = json.loads(entities)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        result.append((str(item[0]), str(item[1])))
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return []


def _parse_relations(relations) -> List[Tuple[str, str, str]]:
    """Parse relations from DSPy structured output (List[Tuple] veya string)."""
    if not relations:
        return []
    
    # Eğer zaten list/tuple ise direkt kullan
    if isinstance(relations, list):
        result = []
        for item in relations:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                result.append((str(item[0]), str(item[1]), str(item[2])))
        return result
    
    # Eğer string ise JSON parse et (backward compatibility)
    if isinstance(relations, str):
        try:
            parsed = json.loads(relations)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        result.append((str(item[0]), str(item[1]), str(item[2])))
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return []


def kg_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Metric function for evaluating KG extraction quality.
    
    Compares predicted entities and relations against gold standard.
    Returns a score between 0.0 and 1.0.
    """
    # Parse gold standard
    gold_entities = _parse_entities(gold.entities) if hasattr(gold, 'entities') else []
    gold_relations = _parse_relations(gold.relations) if hasattr(gold, 'relations') else []
    
    # Parse predictions
    pred_entities = _parse_entities(pred.entities) if hasattr(pred, 'entities') else []
    pred_relations = _parse_relations(pred.relations) if hasattr(pred, 'relations') else []
    
    # Calculate entity F1
    gold_entity_set = set(gold_entities)
    pred_entity_set = set(pred_entities)
    
    if len(gold_entity_set) == 0 and len(pred_entity_set) == 0:
        entity_precision = 1.0
        entity_recall = 1.0
    elif len(pred_entity_set) == 0:
        entity_precision = 0.0
        entity_recall = 0.0
    else:
        entity_precision = len(gold_entity_set & pred_entity_set) / len(pred_entity_set)
        entity_recall = len(gold_entity_set & pred_entity_set) / len(gold_entity_set) if len(gold_entity_set) > 0 else 0.0
    
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
    
    # Calculate relation F1
    gold_relation_set = set(gold_relations)
    pred_relation_set = set(pred_relations)
    
    if len(gold_relation_set) == 0 and len(pred_relation_set) == 0:
        relation_precision = 1.0
        relation_recall = 1.0
    elif len(pred_relation_set) == 0:
        relation_precision = 0.0
        relation_recall = 0.0
    else:
        relation_precision = len(gold_relation_set & pred_relation_set) / len(pred_relation_set)
        relation_recall = len(gold_relation_set & pred_relation_set) / len(gold_relation_set) if len(gold_relation_set) > 0 else 0.0
    
    relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall) if (relation_precision + relation_recall) > 0 else 0.0
    
    # Combined score (weighted average: 40% entities, 60% relations)
    combined_score = 0.4 * entity_f1 + 0.6 * relation_f1
    
    return combined_score


def optimize_extractor(
    extractor: KGExtractor,
    training_examples: List[dspy.Example],
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16
) -> KGExtractor:
    """
    Optimize a KGExtractor using DSPy BootstrapFewShot optimizer.
    
    Args:
        extractor: KGExtractor instance to optimize
        training_examples: List of dspy.Example objects with 'text', 'entities', 'relations' fields
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        max_labeled_demos: Maximum number of labeled demonstrations
    
    Returns:
        Optimized KGExtractor instance
    """
    # Create optimizer
    optimizer = BootstrapFewShot(
        metric=kg_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos
    )
    
    # Optimize the extractor
    optimized_extractor = optimizer.compile(
        student=extractor,
        trainset=training_examples
    )
    
    return optimized_extractor


def refine_triples(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Basic refinement: remove duplicates.
    
    For more advanced refinement, use optimize_extractor with training examples.
    """
    return list(dict.fromkeys(triples))


def merge_entities(
    entities: List[Tuple[str, str]],
    similarity_threshold: float = 0.8
) -> List[Tuple[str, str]]:
    """
    Merge similar entities (e.g., "Apple Inc" and "Apple").
    
    Args:
        entities: List of (name, type) tuples
        similarity_threshold: Threshold for entity name similarity (0.0-1.0)
    
    Returns:
        Merged entities list
    """
    if not entities:
        return []
    
    # Simple exact match merging for now
    # TODO: Add fuzzy matching with similarity_threshold
    seen = {}
    merged = []
    
    for name, etype in entities:
        name_lower = name.lower().strip()
        if name_lower not in seen:
            seen[name_lower] = (name, etype)
            merged.append((name, etype))
        else:
            # Keep the longer/more canonical name
            existing_name, existing_type = seen[name_lower]
            if len(name) > len(existing_name):
                # Replace with longer name
                merged.remove((existing_name, existing_type))
                merged.append((name, etype))
                seen[name_lower] = (name, etype)
    
    return merged


def merge_relations(
    triples: List[Tuple[str, str, str]],
    entity_mapping: Optional[Dict[str, str]] = None
) -> List[Tuple[str, str, str]]:
    """
    Merge relations after entity merging.
    
    Args:
        triples: List of (source, relation, target) tuples
        entity_mapping: Optional mapping from old entity names to new merged names
    
    Returns:
        Merged relations list
    """
    if not triples:
        return []
    
    if entity_mapping:
        # Apply entity mapping
        mapped_triples = []
        for s, r, o in triples:
            s_new = entity_mapping.get(s, s)
            o_new = entity_mapping.get(o, o)
            mapped_triples.append((s_new, r, o_new))
        triples = mapped_triples
    
    # Remove duplicates
    return list(dict.fromkeys(triples))
