"""
Entity Resolution Module

Handles entity normalization and resolution to merge entities that refer to the same real-world object
but appear with different names. Works for ANY domain - technology, business, science, medicine, etc.

Examples:
    - "Dr. Elena Vasquez" and "Elena" → Same person (substring match)
    - "Organization A" and "Org A" → Same organization (abbreviation)
    - "Company X Inc." and "Company X" → Same company (suffix removal)
    - "John Smith" and "J. Smith" → Same person (initial + last name)

This is a critical component for knowledge graph quality - without entity resolution,
the same entity appears as multiple disconnected nodes in the graph.

Supports both string-based and embedding-based similarity for more accurate resolution.
Embedding-based similarity is especially powerful for matching entities with different name variations.

Key features:
    - Adaptive threshold: Shorter names (e.g., "Elena") get lower threshold for better recall
    - Substring matching: Aggressive boosting for substring matches (e.g., "Elena" in "Dr. Elena Vasquez")
    - Name normalization: Removes titles, suffixes (Dr., Inc., etc.) for better matching
    - Hybrid similarity: Combines string and embedding similarity for best accuracy
    - Domain-agnostic: Works with any entity types (Person, Company, Product, Location, etc.)
"""
from typing import List, Tuple, Dict, Set, Optional, Any
import logging
import re
from difflib import SequenceMatcher
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for embedding provider
_EmbeddingProvider = None


def _get_embedding_provider():
    """Lazy import of EmbeddingProvider."""
    global _EmbeddingProvider
    if _EmbeddingProvider is None:
        try:
            from ..embedding import EmbeddingProvider
            _EmbeddingProvider = EmbeddingProvider
        except ImportError:
            pass
    return _EmbeddingProvider


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors (0.0 to 1.0).
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector dimensions must match: {len(vec1)} != {len(vec2)}")
    
    # Convert to numpy arrays for efficient computation
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Calculate cosine similarity: dot product / (norm1 * norm2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    # Normalize to 0.0-1.0 range (cosine similarity is already in -1.0 to 1.0, but for embeddings it's typically 0.0-1.0)
    return float(max(0.0, min(1.0, similarity)))


def similarity_score(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings (0.0 to 1.0).
    
    Uses SequenceMatcher for better performance on common cases.
    Also checks for substring/superset relationships (e.g., "Elena" in "Dr. Elena Vasquez").
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
    
    # Safety rule: never merge two different single-token names by string similarity alone.
    # Example: "Elena" vs "Selena" are very similar strings but should not be merged.
    # If these need to be merged, it should happen via explicit alias rules or embeddings with strong evidence.
    if (" " not in s1) and (" " not in s2) and s1 != s2:
        return 0.0

    # Token-boundary containment (important for short aliases)
    # "elena" should match "elena vasquez" but NOT "selena".
    def _word_boundary_contains(short: str, long: str) -> bool:
        if len(short) < 3:
            return False
        return re.search(rf"(?i)(?<!\w){re.escape(short)}(?!\w)", long) is not None

    if _word_boundary_contains(s1, s2) or _word_boundary_contains(s2, s1):
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        base_score = shorter / longer if longer else 0.0
        # For boundary-contained aliases we boost aggressively but conservatively.
        # Minimum 0.75 for meaningful alias containment.
        boosted_score = max(0.75, min(0.95, 0.75 + base_score * 0.25))
        seq_similarity = SequenceMatcher(None, s1, s2).ratio()
        return max(boosted_score, seq_similarity)
    
    # Use SequenceMatcher for similarity (standard edit distance)
    return SequenceMatcher(None, s1, s2).ratio()


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for comparison (case-insensitive).
    
    This function handles case sensitivity issues by converting to lowercase
    and normalizing common prefixes/titles/suffixes that don't affect entity identity.
    Works for ANY domain - removes common titles across languages and domains.
    
    Examples:
        - "Cognitive Enhancement" → "cognitive enhancement"
        - "cognitive enhancement" → "cognitive enhancement"
        - "Dr. Elena Vasquez" → "elena vasquez" (title removed)
        - "Dr Vasquez" → "vasquez"
        - "Prof. John Smith" → "john smith"
        - "Company X Inc." → "company x" (suffix removed)
        - "Organization A Ltd." → "organization a" (suffix removed)
    
    Args:
        name: Entity name to normalize (any domain, any language)
    
    Returns:
        Normalized name (lowercase, whitespace normalized, titles/suffixes removed)
    """
    # Lowercase, strip, remove extra spaces
    normalized = name.lower().strip()
    
    # Remove common titles/prefixes that don't affect identity
    # This helps match "Dr. Elena Vasquez" with "Elena Vasquez" or "Elena"
    title_patterns = [
        r'^(dr|doctor|prof|professor|mr|mrs|miss|ms|sir|madam|lord|lady)\s*\.?\s*',
        r'\s+(jr|sr|jr\.|sr\.|ii|iii|iv)$',  # Suffixes
    ]
    for pattern in title_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # Normalize whitespace (multiple spaces -> single space)
    normalized = " ".join(normalized.split())
    return normalized


class EntityResolver:
    """Entity resolver for merging duplicate entity references.
    
    This module handles entity resolution (merging entities with different names).
    For pronoun and reference resolution, see drg.coreference_resolution module.
    
    Supports both string-based and embedding-based similarity:
    - String similarity: Fast, works without dependencies
    - Embedding similarity: More accurate, especially for name variations (e.g., "Dr. Elena Vasquez" vs "Elena")
    
    Uses hybrid approach: if embedding provider is available, uses both methods and combines scores.
    Falls back to string similarity if embeddings are not available.
    
    Note: Coreference resolution (pronouns, references) is handled separately
    in drg.coreference_resolution module and should be applied BEFORE entity resolution.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.65,
        use_normalization: bool = True,
        adaptive_threshold: bool = True,
        embedding_provider = None,
        use_embedding: bool = True,
        embedding_weight: float = 0.7,
        min_merge_margin: float = 0.08,
    ):
        """Initialize entity resolver.
        
        Args:
            similarity_threshold: Base similarity score threshold (0.0-1.0).
                                 Default 0.65 (lowered from 0.85) for better recall on short names.
            use_normalization: Whether to use name normalization before comparison
            adaptive_threshold: Whether to use adaptive threshold based on name length.
                               Shorter names get lower threshold (e.g., "Elena" vs "Dr. Elena Vasquez").
            embedding_provider: Optional EmbeddingProvider for embedding-based similarity.
                               If None and use_embedding=True, will try to create a default provider.
            use_embedding: Whether to use embedding-based similarity if available (default: True).
                          Falls back to string similarity if embeddings are not available.
            embedding_weight: Weight for embedding similarity when combining with string similarity (0.0-1.0).
                             Higher values prioritize embedding similarity. Default: 0.7.
                             Combined score = embedding_weight * embedding_sim + (1 - embedding_weight) * string_sim
        """
        self.base_similarity_threshold = similarity_threshold
        self.similarity_threshold = similarity_threshold  # Will be adjusted adaptively
        self.use_normalization = use_normalization
        self.adaptive_threshold = adaptive_threshold
        self.use_embedding = use_embedding
        self.embedding_weight = embedding_weight
        # Conservative gating: require best match to be meaningfully better than second best.
        self.min_merge_margin = min_merge_margin
        
        # Setup embedding provider
        self.embedding_provider = None
        if use_embedding:
            if embedding_provider is not None:
                EmbeddingProviderClass = _get_embedding_provider()
                if EmbeddingProviderClass and isinstance(embedding_provider, EmbeddingProviderClass):
                    self.embedding_provider = embedding_provider
                    logger.info(f"Using embedding provider: {embedding_provider.get_model_name()}")
                else:
                    logger.warning("Provided embedding_provider is not a valid EmbeddingProvider, falling back to string similarity")
            else:
                # Try to create default embedding provider (local if available)
                try:
                    from ..embedding import create_embedding_provider
                    try:
                        self.embedding_provider = create_embedding_provider("local")
                        logger.info(f"Using default local embedding provider: {self.embedding_provider.get_model_name()}")
                    except (ImportError, ValueError):
                        # Local not available, try others
                        logger.debug("Local embedding provider not available, entity resolution will use string similarity only")
                except ImportError:
                    logger.debug("Embedding module not available, entity resolution will use string similarity only")
        
        # Cache for entity embeddings to avoid recomputation
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def _get_adaptive_threshold(self, name1: str, name2: str) -> float:
        """Calculate adaptive threshold based on name lengths.
        
        Shorter names need lower threshold (more lenient matching).
        Long names can use higher threshold (more strict matching).
        Also considers substring relationships for better recall.
        """
        if not self.adaptive_threshold:
            return self.base_similarity_threshold
        
        min_len = min(len(name1), len(name2))
        max_len = max(len(name1), len(name2))
        
        # Check for substring relationship (strong indicator of same entity)
        # Normalize names for better substring detection
        s1_normalized = normalize_entity_name(name1) if self.use_normalization else name1.lower().strip()
        s2_normalized = normalize_entity_name(name2) if self.use_normalization else name2.lower().strip()
        
        if (s1_normalized in s2_normalized or s2_normalized in s1_normalized) and min_len >= 3:
            # Substring match with at least 3 chars → very lenient threshold
            # "Elena" in "elena vasquez" (after normalization) → should match
            # Use even more lenient threshold for substring matches (0.30 instead of 0.40)
            # because substring matching in similarity_score already returns high scores (0.70+)
            return 0.30  # Very lenient for substring matches (since similarity_score already boosts them)
        
        # If one name is much shorter, use lower threshold
        if min_len < 5 and max_len > 10:
            return max(0.40, self.base_similarity_threshold - 0.25)  # Very lenient for short names
        elif min_len < 8:
            return max(0.50, self.base_similarity_threshold - 0.15)  # Lenient for short names
        elif min_len < 10:
            return max(0.55, self.base_similarity_threshold - 0.10)  # Slightly lenient for medium names
        
        # For long names, use base threshold
        return self.base_similarity_threshold
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names using hybrid approach.
        
        Uses both string similarity and embedding similarity if available.
        Also applies normalization before comparison.
        
        Args:
            name1: First entity name
            name2: Second entity name
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize names if enabled
        if self.use_normalization:
            norm1 = normalize_entity_name(name1)
            norm2 = normalize_entity_name(name2)
        else:
            norm1 = name1.lower().strip()
            norm2 = name2.lower().strip()
        
        # Calculate string similarity (always available)
        string_sim = similarity_score(norm1, norm2)
        
        # If embeddings available, calculate embedding similarity and combine
        if self.embedding_provider and self.use_embedding:
            try:
                # Get embeddings (use cache if available)
                emb1 = self._get_entity_embedding(name1)
                emb2 = self._get_entity_embedding(name2)
                
                if emb1 and emb2:
                    embedding_sim = cosine_similarity(emb1, emb2)
                    
                    # Combine string and embedding similarity
                    # Embedding similarity is often better for name variations
                    combined_sim = (
                        self.embedding_weight * embedding_sim +
                        (1.0 - self.embedding_weight) * string_sim
                    )
                    
                    # For substring matches, boost the combined score
                    # (Embedding similarity might miss exact substring matches)
                    if norm1 in norm2 or norm2 in norm1:
                        min_len = min(len(norm1), len(norm2))
                        max_len = max(len(norm1), len(norm2))
                        if min_len >= 3 and min_len / max_len > 0.2:
                            # Substring match detected - boost combined score
                            combined_sim = max(combined_sim, 0.75)  # At least 0.75 for meaningful substring matches
                    
                    return combined_sim
            except Exception as e:
                logger.debug(f"Embedding similarity calculation failed: {e}, using string similarity only")
        
        # Fallback to string similarity
        return string_sim
    
    def _get_entity_embedding(self, name: str) -> Optional[List[float]]:
        """Get embedding for an entity name, using cache if available.
        
        Args:
            name: Entity name
        
        Returns:
            Embedding vector or None if not available
        """
        # Use normalized name for cache key
        cache_key = normalize_entity_name(name) if self.use_normalization else name.lower().strip()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if not self.embedding_provider:
            return None
        
        try:
            embedding = self.embedding_provider.embed(name)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.debug(f"Failed to get embedding for '{name}': {e}")
            return None
    
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

        # Detect ambiguous single-token aliases (safe, input-agnostic):
        # If a singleton token (e.g., "elena") appears in 2+ multi-token names of the same type,
        # do NOT auto-merge it — we prefer abstaining over guessing.
        normalized_names = [
            (name, normalize_entity_name(name) if self.use_normalization else name.lower().strip())
            for name, _ in entities
        ]
        multi_token_norms = [n for _, n in normalized_names if len(n.split()) >= 2]
        ambiguous_singletons: Set[str] = set()
        for orig, norm in normalized_names:
            if len(norm.split()) != 1:
                continue
            token = norm
            hits = 0
            for long_norm in multi_token_norms:
                long_tokens = set(long_norm.split())
                if token in long_tokens:
                    hits += 1
                    if hits >= 2:
                        ambiguous_singletons.add(token)
                        break

        def _safe_to_merge(n1: str, n2: str, et: str) -> bool:
            """Conservative merge gating to keep entity resolution safe across arbitrary inputs."""
            a = normalize_entity_name(n1) if self.use_normalization else n1.lower().strip()
            b = normalize_entity_name(n2) if self.use_normalization else n2.lower().strip()
            if a == b:
                return True
            ta = a.split()
            tb = b.split()
            # Never merge two different single-token names.
            if len(ta) == 1 and len(tb) == 1:
                return False
            # Never auto-merge ambiguous single-token aliases into multi-token names.
            if et.lower() == "person":
                if len(ta) == 1 and len(tb) >= 2 and ta[0] in set(tb) and ta[0] in ambiguous_singletons:
                    return False
                if len(tb) == 1 and len(ta) >= 2 and tb[0] in set(ta) and tb[0] in ambiguous_singletons:
                    return False
            # Person names: require last-name agreement when both are multi-token.
            if et.lower() == "person" and len(ta) >= 2 and len(tb) >= 2:
                return ta[-1] == tb[-1]
            return True
        
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
            
            # Conservative best-match selection (avoid "first match wins" for short/ambiguous aliases).
            best = None
            best_sim = -1.0
            second_sim = -1.0
            for canonical in canonical_groups.keys():
                if not _safe_to_merge(name, canonical, etype):
                    continue
                sim = self._calculate_similarity(name, canonical)
                thr = self._get_adaptive_threshold(name, canonical)
                if sim >= thr:
                    if sim > best_sim:
                        second_sim = best_sim
                        best_sim = sim
                        best = canonical
                    elif sim > second_sim:
                        second_sim = sim

            if best is not None and (
                second_sim < 0
                or (best_sim - second_sim) >= self.min_merge_margin
            ):
                matched_canonical = best
            
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
                    if not _safe_to_merge(name, other_name, etype):
                        continue
                    
                    # Use hybrid similarity (embedding + string)
                    similarity = self._calculate_similarity(name, other_name)
                    
                    # Use adaptive threshold
                    adaptive_threshold = self._get_adaptive_threshold(name, other_name)
                    
                    if similarity >= adaptive_threshold:
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
    similarity_threshold: float = 0.65,
    adaptive_threshold: bool = True,
    embedding_provider: Optional[Any] = None,
    use_embedding: bool = True,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Convenience function to resolve both entities and relations.
    
    Args:
        entities: List of (entity_name, entity_type) tuples
        relations: List of (source, relation, target) tuples
        similarity_threshold: Minimum similarity score for entity resolution
        adaptive_threshold: Whether to use adaptive threshold based on name length
        embedding_provider: Optional EmbeddingProvider for embedding-based similarity
        use_embedding: Whether to use embedding-based similarity if available (default: True)
    
    Returns:
        Tuple of (resolved_entities, resolved_relations)
    """
    resolver = EntityResolver(
        similarity_threshold=similarity_threshold,
        adaptive_threshold=adaptive_threshold,
        embedding_provider=embedding_provider,
        use_embedding=use_embedding,
    )
    resolved_entities, name_mapping = resolver.resolve(entities)
    resolved_relations = resolver.resolve_relations(relations, name_mapping)
    
    return resolved_entities, resolved_relations

