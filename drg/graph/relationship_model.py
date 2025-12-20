"""
Enriched Relationship Model - GraphRAG Format

This module provides enriched relationship modeling with:
- Structured relationship representation (source, target, type, detail, confidence, source_ref)
- Relationship type taxonomy
- Rule-based classifier (type compatibility + schema constraints)
- LLM-based classifier stub (for future implementation)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum


# Relationship Type Taxonomy
class RelationshipType(str, Enum):
    """
    Standard relationship type taxonomy.
    These types are domain-agnostic and can be used across different datasets.
    """
    # Causal relationships
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    TRIGGERS = "triggers"
    RESULTS_IN = "results_in"
    
    # Spatial relationships
    LOCATED_AT = "located_at"
    CONTAINS = "contains"
    NEAR = "near"
    INSIDE = "inside"
    OUTSIDE = "outside"
    
    # Temporal relationships
    OCCURS_BEFORE = "occurs_before"
    OCCURS_AFTER = "occurs_after"
    OCCURS_DURING = "occurs_during"
    FOLLOWS = "follows"
    
    # Social/Interaction relationships
    INFLUENCES = "influences"
    INFLUENCED_BY = "influenced_by"
    COLLABORATES_WITH = "collaborates_with"
    WORKS_WITH = "works_with"
    OWNS = "owns"
    BELONGS_TO = "belongs_to"
    MEMBER_OF = "member_of"
    
    # Hierarchical relationships
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    
    # Similarity/Equivalence relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    EQUIVALENT_TO = "equivalent_to"
    
    # Action relationships
    CREATES = "creates"
    DESTROYS = "destroys"
    MODIFIES = "modifies"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    
    # Communication relationships
    COMMUNICATES_WITH = "communicates_with"
    INFORMS = "informs"
    REQUESTS = "requests"
    RESPONDS_TO = "responds_to"
    
    # Emotional/Subjective relationships
    LIKES = "likes"
    DISLIKES = "dislikes"
    LOVES = "loves"
    HATES = "hates"
    FEARS = "fears"
    TRUSTS = "trusts"
    
    # Other common relationships
    KNOWS = "knows"
    MEETS = "meets"
    VISITS = "visits"
    LEAVES = "leaves"
    RETURNS_TO = "returns_to"


# Relationship type categories for better organization
RELATIONSHIP_CATEGORIES = {
    "causal": [
        RelationshipType.CAUSES,
        RelationshipType.CAUSED_BY,
        RelationshipType.TRIGGERS,
        RelationshipType.RESULTS_IN,
    ],
    "spatial": [
        RelationshipType.LOCATED_AT,
        RelationshipType.CONTAINS,
        RelationshipType.NEAR,
        RelationshipType.INSIDE,
        RelationshipType.OUTSIDE,
    ],
    "temporal": [
        RelationshipType.OCCURS_BEFORE,
        RelationshipType.OCCURS_AFTER,
        RelationshipType.OCCURS_DURING,
        RelationshipType.FOLLOWS,
    ],
    "social": [
        RelationshipType.INFLUENCES,
        RelationshipType.INFLUENCED_BY,
        RelationshipType.COLLABORATES_WITH,
        RelationshipType.WORKS_WITH,
        RelationshipType.OWNS,
        RelationshipType.BELONGS_TO,
        RelationshipType.MEMBER_OF,
    ],
    "hierarchical": [
        RelationshipType.PARENT_OF,
        RelationshipType.CHILD_OF,
        RelationshipType.PART_OF,
        RelationshipType.HAS_PART,
    ],
    "action": [
        RelationshipType.CREATES,
        RelationshipType.DESTROYS,
        RelationshipType.MODIFIES,
        RelationshipType.PRODUCES,
        RelationshipType.CONSUMES,
    ],
    "communication": [
        RelationshipType.COMMUNICATES_WITH,
        RelationshipType.INFORMS,
        RelationshipType.REQUESTS,
        RelationshipType.RESPONDS_TO,
    ],
    "emotional": [
        RelationshipType.LIKES,
        RelationshipType.DISLIKES,
        RelationshipType.LOVES,
        RelationshipType.HATES,
        RelationshipType.FEARS,
        RelationshipType.TRUSTS,
    ],
}


@dataclass
class EnrichedRelationship:
    """
    GraphRAG-format enriched relationship representation.
    
    Each relationship includes:
    - source: Source entity identifier
    - target: Target entity identifier
    - relationship_type: Type from RelationshipType taxonomy
    - relationship_detail: Short descriptive sentence explaining the relationship
    - confidence: Confidence score (0.0 to 1.0)
    - source_ref: Reference to source (e.g., chunk_id, document_id) - placeholder for now
    """
    source: str
    target: str
    relationship_type: RelationshipType
    relationship_detail: str
    confidence: float = 1.0
    source_ref: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate relationship data."""
        if not self.source:
            raise ValueError("Source cannot be empty")
        if not self.target:
            raise ValueError("Target cannot be empty")
        if not self.relationship_detail:
            raise ValueError("Relationship detail cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.source == self.target:
            raise ValueError("Source and target cannot be the same")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type.value,
            "relationship_detail": self.relationship_detail,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedRelationship":
        """Create from dictionary representation."""
        return cls(
            source=data["source"],
            target=data["target"],
            relationship_type=RelationshipType(data["relationship_type"]),
            relationship_detail=data["relationship_detail"],
            confidence=data.get("confidence", 1.0),
            source_ref=data.get("source_ref"),
        )
    
    def to_graphrag_format(self) -> Dict[str, Any]:
        """Convert to GraphRAG-compatible format."""
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type.value,
            "relationship_detail": self.relationship_detail,
            "confidence": self.confidence,
            "source_ref": self.source_ref or "unknown",
        }


class RelationshipTypeClassifier:
    """
    Classifier for determining relationship types from raw relationships.
    Combines rule-based and LLM-based approaches.
    """
    
    def __init__(self, schema: Optional[Any] = None):
        """
        Initialize classifier.
        
        Args:
            schema: Optional schema object (EnhancedDRGSchema or DRGSchema)
                   for constraint checking
        """
        self.schema = schema
        self._build_schema_indexes()
    
    def _build_schema_indexes(self):
        """Build indexes from schema for fast lookup."""
        if self.schema is None:
            self._valid_relations: Dict[Tuple[str, str], Set[str]] = {}
            return
        
        # Extract valid (source_type, target_type) -> relationship_types mapping
        self._valid_relations: Dict[Tuple[str, str], Set[str]] = {}
        
        try:
            # Try EnhancedDRGSchema
            if hasattr(self.schema, 'relation_groups'):
                for rg in self.schema.relation_groups:
                    for rel in rg.relations:
                        key = (rel.src, rel.dst)
                        if key not in self._valid_relations:
                            self._valid_relations[key] = set()
                        self._valid_relations[key].add(rel.name.lower())
        except AttributeError:
            # Try legacy DRGSchema
            if hasattr(self.schema, 'relations'):
                for rel in self.schema.relations:
                    key = (rel.src, rel.dst)
                    if key not in self._valid_relations:
                        self._valid_relations[key] = set()
                    self._valid_relations[key].add(rel.name.lower())
    
    def classify(
        self,
        source: str,
        target: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
        raw_relation_text: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Tuple[RelationshipType, float]]:
        """
        Classify relationship type(s) with confidence scores.
        
        Args:
            source: Source entity identifier
            target: Target entity identifier
            source_type: Optional source entity type
            target_type: Optional target entity type
            raw_relation_text: Optional raw text describing the relationship
            context: Optional context text
        
        Returns:
            List of (RelationshipType, confidence) tuples, sorted by confidence
        """
        # First, try rule-based classification
        rule_based_results = self._classify_rule_based(
            source_type=source_type,
            target_type=target_type,
            raw_relation_text=raw_relation_text,
        )
        
        # If we have schema constraints, filter results
        if self.schema and source_type and target_type:
            rule_based_results = self._apply_schema_constraints(
                rule_based_results,
                source_type=source_type,
                target_type=target_type,
            )
        
        # If rule-based gives high confidence, return it
        if rule_based_results and rule_based_results[0][1] >= 0.8:
            return rule_based_results[:3]  # Top 3
        
        # Otherwise, try LLM-based (stub for now)
        llm_results = self._classify_llm_based(
            source=source,
            target=target,
            source_type=source_type,
            target_type=target_type,
            raw_relation_text=raw_relation_text,
            context=context,
        )
        
        # Combine and sort by confidence
        all_results = rule_based_results + llm_results
        # Remove duplicates, keeping highest confidence
        seen_types = {}
        for rel_type, conf in all_results:
            if rel_type not in seen_types or conf > seen_types[rel_type]:
                seen_types[rel_type] = conf
        
        results = sorted(seen_types.items(), key=lambda x: x[1], reverse=True)
        return results[:5]  # Top 5
    
    def _classify_rule_based(
        self,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
        raw_relation_text: Optional[str] = None,
    ) -> List[Tuple[RelationshipType, float]]:
        """
        Rule-based classification using pattern matching and type compatibility.
        
        Returns:
            List of (RelationshipType, confidence) tuples
        """
        results = []
        
        # Pattern matching on raw_relation_text if available
        if raw_relation_text:
            text_lower = raw_relation_text.lower()
            
            # Causal patterns
            if any(word in text_lower for word in ["causes", "caused", "leads to", "results in"]):
                results.append((RelationshipType.CAUSES, 0.9))
            if any(word in text_lower for word in ["because", "due to", "result of"]):
                results.append((RelationshipType.CAUSED_BY, 0.9))
            
            # Spatial patterns
            if any(word in text_lower for word in ["located", "at", "in", "place"]):
                results.append((RelationshipType.LOCATED_AT, 0.8))
            if any(word in text_lower for word in ["contains", "includes", "has"]):
                results.append((RelationshipType.CONTAINS, 0.8))
            
            # Temporal patterns
            if any(word in text_lower for word in ["before", "prior", "earlier"]):
                results.append((RelationshipType.OCCURS_BEFORE, 0.8))
            if any(word in text_lower for word in ["after", "later", "subsequent"]):
                results.append((RelationshipType.OCCURS_AFTER, 0.8))
            
            # Social patterns
            if any(word in text_lower for word in ["influences", "affects", "impacts"]):
                results.append((RelationshipType.INFLUENCES, 0.85))
            if any(word in text_lower for word in ["collaborates", "works with", "partners"]):
                results.append((RelationshipType.COLLABORATES_WITH, 0.85))
            if any(word in text_lower for word in ["owns", "possesses", "has ownership"]):
                results.append((RelationshipType.OWNS, 0.9))
            if any(word in text_lower for word in ["member", "belongs", "part of group"]):
                results.append((RelationshipType.MEMBER_OF, 0.85))
            
            # Hierarchical patterns
            if any(word in text_lower for word in ["parent", "father", "mother"]):
                results.append((RelationshipType.PARENT_OF, 0.9))
            if any(word in text_lower for word in ["child", "son", "daughter"]):
                results.append((RelationshipType.CHILD_OF, 0.9))
            if any(word in text_lower for word in ["part of", "component", "belongs to"]):
                results.append((RelationshipType.PART_OF, 0.8))
        
        # Type-based heuristics (if entity types are known)
        if source_type and target_type:
            # Person -> Person: likely social relationships
            if source_type == "Person" and target_type == "Person":
                if not results:  # Only if no text-based match
                    results.append((RelationshipType.RELATED_TO, 0.5))
                    results.append((RelationshipType.KNOWS, 0.4))
            
            # Person -> Location: likely spatial
            elif source_type == "Person" and target_type == "Location":
                if not results:
                    results.append((RelationshipType.LOCATED_AT, 0.6))
                    results.append((RelationshipType.VISITS, 0.5))
            
            # Event -> Person: likely involves/influences
            elif source_type == "Event" and target_type == "Person":
                if not results:
                    results.append((RelationshipType.INFLUENCES, 0.5))
                    results.append((RelationshipType.RESULTS_IN, 0.4))
        
        # Default fallback
        if not results:
            results.append((RelationshipType.RELATED_TO, 0.3))
        
        return results
    
    def _apply_schema_constraints(
        self,
        candidates: List[Tuple[RelationshipType, float]],
        source_type: str,
        target_type: str,
    ) -> List[Tuple[RelationshipType, float]]:
        """
        Filter candidates based on schema constraints.
        
        Args:
            candidates: List of (RelationshipType, confidence) tuples
            source_type: Source entity type
            target_type: Target entity type
        
        Returns:
            Filtered list of candidates
        """
        if not self.schema:
            return candidates
        
        key = (source_type, target_type)
        valid_relation_names = self._valid_relations.get(key, set())
        
        # If no schema constraints for this pair, return all candidates
        if not valid_relation_names:
            return candidates
        
        # Filter candidates to only include schema-valid ones
        # Map RelationshipType enum values to schema relation names
        filtered = []
        for rel_type, conf in candidates:
            rel_name = rel_type.value.lower()
            # Check if exact match or similar (fuzzy matching)
            if rel_name in valid_relation_names:
                filtered.append((rel_type, conf))
            # Also check for partial matches (e.g., "causes" matches "caused_by")
            elif any(rel_name in valid or valid in rel_name for valid in valid_relation_names):
                filtered.append((rel_type, conf * 0.8))  # Lower confidence for partial match
        
        return filtered if filtered else candidates  # Return original if no matches
    
    def _classify_llm_based(
        self,
        source: str,
        target: str,
        source_type: Optional[str] = None,
        target_type: Optional[str] = None,
        raw_relation_text: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Tuple[RelationshipType, float]]:
        """
        LLM-based classification (STUB - to be implemented later).
        
        This is a placeholder that returns empty results.
        Future implementation will use LLM to classify relationship types
        based on context and entity information.
        
        Returns:
            Empty list (stub implementation)
        """
        # TODO: Implement LLM-based classification
        # This would use DSPy or similar to:
        # 1. Create a prompt with source, target, context
        # 2. Ask LLM to classify relationship type
        # 3. Extract relationship detail explanation
        # 4. Return classified type with confidence
        return []


def create_enriched_relationship(
    source: str,
    target: str,
    relationship_type: RelationshipType,
    relationship_detail: str,
    confidence: float = 1.0,
    source_ref: Optional[str] = None,
) -> EnrichedRelationship:
    """
    Factory function to create an EnrichedRelationship.
    
    Args:
        source: Source entity identifier
        target: Target entity identifier
        relationship_type: Relationship type from taxonomy
        relationship_detail: Short descriptive sentence
        confidence: Confidence score (0.0 to 1.0)
        source_ref: Optional source reference (e.g., chunk_id)
    
    Returns:
        EnrichedRelationship instance
    """
    return EnrichedRelationship(
        source=source,
        target=target,
        relationship_type=relationship_type,
        relationship_detail=relationship_detail,
        confidence=confidence,
        source_ref=source_ref,
    )





