"""
Entity Resolution Module

Handles entity normalization and resolution to merge entities that refer to the same real-world object
but appear with different names (e.g., "Dr. Elena Vasquez", "Dr. Vasquez", "Dr. Elena").

This is a critical component for knowledge graph quality - without entity resolution,
the same entity appears as multiple disconnected nodes in the graph.
"""
from typing import List, Tuple, Dict, Set, Optional
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings (0.0 to 1.0).
    
    Uses SequenceMatcher for better performance on common cases.
    For more sophisticated matching, consider using fuzzywuzzy or rapidfuzz.
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize strings
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, s1, s2).ratio()


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison (case-insensitive).
    
    This function handles case sensitivity issues by converting to lowercase
    and normalizing common prefixes/titles (e.g., "Dr.", "Dr", "dr").
    
    Examples:
        - "Cognitive Enhancement" → "cognitive enhancement"
        - "cognitive enhancement" → "cognitive enhancement"
        - "Dr. Elena Vasquez" → "dr elena vasquez"
        - "Dr Vasquez" → "dr vasquez"
    
    Args:
        name: Entity name to normalize
    
    Returns:
        Normalized name (lowercase, whitespace normalized)
    """
    # Lowercase, strip, remove extra spaces
    normalized = name.lower().strip()
    # Normalize common prefixes/titles that don't affect identity
    # Handle variations: "Dr.", "Dr", "dr", "dr."
    normalized = normalized.replace("dr.", "dr ").replace("dr.", "dr ")
    # Normalize whitespace (multiple spaces -> single space)
    normalized = " ".join(normalized.split())
    return normalized


class EntityResolver:
    """Entity resolver for merging duplicate entity references.
    
    This module handles entity resolution (merging entities with different names).
    For pronoun and reference resolution, see drg.coreference_resolution module.
    
    Basic implementation using string similarity and normalization.
    For production use, consider more sophisticated approaches:
    - Fuzzy matching libraries (rapidfuzz, fuzzywuzzy)
    - Embedding-based similarity
    - Context-aware resolution
    
    Note: Coreference resolution (pronouns, references) is handled separately
    in drg.coreference_resolution module and should be applied BEFORE entity resolution.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, use_normalization: bool = True):
        """Initialize entity resolver.
        
        Args:
            similarity_threshold: Minimum similarity score to consider entities as the same (0.0-1.0)
            use_normalization: Whether to use name normalization before comparison
        """
        self.similarity_threshold = similarity_threshold
        self.use_normalization = use_normalization
    
    def resolve(
        self,
        entities: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """Resolve duplicate entity references.
        
        Args:
            entities: List of (entity_name, entity_type) tuples
        
        Returns:
            Tuple of (resolved_entities, name_mapping) where:
            - resolved_entities: List of unique (entity_name, entity_type) tuples
            - name_mapping: Dict mapping original names to canonical names
        """
        if not entities:
            return [], {}
        
        # Group entities by type for better resolution (same type entities are more likely duplicates)
        entities_by_type: Dict[str, List[Tuple[str, str]]] = {}
        for name, etype in entities:
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append((name, etype))
        
        # Resolve within each entity type
        all_resolved: List[Tuple[str, str]] = []
        name_mapping: Dict[str, str] = {}
        
        for etype, type_entities in entities_by_type.items():
            resolved, mapping = self._resolve_by_type(type_entities)
            all_resolved.extend(resolved)
            name_mapping.update(mapping)
        
        logger.info(
            f"Entity resolution: {len(entities)} entities -> {len(all_resolved)} unique entities "
            f"({len(entities) - len(all_resolved)} duplicates resolved)"
        )
        
        return all_resolved, name_mapping
    
    def _resolve_by_type(
        self,
        entities: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
        """Resolve entities of the same type.
        
        Args:
            entities: List of (entity_name, entity_type) tuples (all same type)
        
        Returns:
            Tuple of (resolved_entities, name_mapping)
        """
        if len(entities) <= 1:
            return entities, {}
        
        # Build canonical name -> all variant names mapping
        canonical_groups: Dict[str, List[str]] = {}
        name_mapping: Dict[str, str] = {}
        processed = set()
        
        for i, (name, etype) in enumerate(entities):
            if name in processed:
                continue
            
            # Try to find a matching canonical name
            normalized_name = normalize_entity_name(name) if self.use_normalization else name.lower()
            matched_canonical = None
            
            for canonical in canonical_groups.keys():
                canonical_normalized = normalize_entity_name(canonical) if self.use_normalization else canonical.lower()
                similarity = similarity_score(normalized_name, canonical_normalized)
                
                if similarity >= self.similarity_threshold:
                    matched_canonical = canonical
                    break
            
            if matched_canonical:
                # Add to existing group
                canonical_groups[matched_canonical].append(name)
                name_mapping[name] = matched_canonical
                processed.add(name)
            else:
                # Create new canonical group (use longest/fullest name as canonical)
                # Find the longest name among similar ones as canonical
                similar_names = [name]
                for j, (other_name, _) in enumerate(entities[i+1:], start=i+1):
                    if other_name in processed:
                        continue
                    
                    other_normalized = normalize_entity_name(other_name) if self.use_normalization else other_name.lower()
                    similarity = similarity_score(normalized_name, other_normalized)
                    
                    if similarity >= self.similarity_threshold:
                        similar_names.append(other_name)
                        processed.add(other_name)
                
                # Use longest name as canonical (usually most complete)
                canonical = max(similar_names, key=len)
                canonical_groups[canonical] = similar_names
                
                for variant in similar_names:
                    name_mapping[variant] = canonical
                    processed.add(variant)
        
        # Build resolved entities list
        etype = entities[0][1]  # All entities have same type
        resolved = [(canonical, etype) for canonical in canonical_groups.keys()]
        
        return resolved, name_mapping
    
    def resolve_relations(
        self,
        relations: List[Tuple[str, str, str]],
        name_mapping: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """Resolve entity names in relations using the name mapping.
        
        Args:
            relations: List of (source, relation, target) tuples
            name_mapping: Mapping from original names to canonical names
        
        Returns:
            List of resolved relations with canonical entity names
        """
        resolved = []
        seen = set()
        
        for s, r, o in relations:
            # Map to canonical names
            canonical_s = name_mapping.get(s, s)
            canonical_o = name_mapping.get(o, o)
            
            # Skip self-relations (entity relates to itself after resolution)
            if canonical_s == canonical_o:
                logger.debug(f"Skipping self-relation: {canonical_s} --{r}--> {canonical_s}")
                continue
            
            # Deduplicate
            triple = (canonical_s, r, canonical_o)
            if triple not in seen:
                resolved.append(triple)
                seen.add(triple)
        
        logger.info(
            f"Relation resolution: {len(relations)} relations -> {len(resolved)} unique relations "
            f"({len(relations) - len(resolved)} duplicates/self-relations removed)"
        )
        
        return resolved


def resolve_entities_and_relations(
    entities: List[Tuple[str, str]],
    relations: List[Tuple[str, str, str]],
    similarity_threshold: float = 0.85
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Convenience function to resolve both entities and relations.
    
    Args:
        entities: List of (entity_name, entity_type) tuples
        relations: List of (source, relation, target) tuples
        similarity_threshold: Minimum similarity score for entity resolution
    
    Returns:
        Tuple of (resolved_entities, resolved_relations)
    """
    resolver = EntityResolver(similarity_threshold=similarity_threshold)
    resolved_entities, name_mapping = resolver.resolve(entities)
    resolved_relations = resolver.resolve_relations(relations, name_mapping)
    
    return resolved_entities, resolved_relations

