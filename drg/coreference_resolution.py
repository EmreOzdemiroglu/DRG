"""
Coreference Resolution Module

Handles pronoun and reference resolution to link pronouns (he, she, it, they) and 
definite noun phrases (the company, this person) to their antecedent entities.

This complements entity resolution by resolving references to entities that appear
as pronouns or noun phrases rather than explicit entity names.

Example:
    "Elon Musk founded Tesla. He is the CEO."
    → "He" → "Elon Musk" (pronoun resolution)
    
    "The company produces cars. It is located in California."
    → "It" → "Tesla" (pronoun resolution, requires entity resolution first)

Integration with Entity Resolution:
    Coreference resolution should be applied BEFORE entity resolution, as it creates
    explicit entity mentions that entity resolution can then merge.
    
    Pipeline:
    1. Extract entities and relations (LLM extraction)
    2. Coreference resolution (pronouns → explicit entities)
    3. Entity resolution (merge duplicate mentions)
    4. Final KG construction
"""

from typing import List, Tuple, Dict, Set, Optional
import logging
import re

logger = logging.getLogger(__name__)


class CoreferenceResolver:
    """Resolves pronouns and noun phrase references to explicit entities.
    
    Basic implementation using simple heuristics. For production use, consider:
    - spaCy with neural coreference resolution
    - NeuralCoref (spaCy extension)
    - AllenNLP coreference resolution
    - HuggingFace Transformers (e.g., coref-roberta-base)
    """
    
    def __init__(
        self,
        use_nlp: bool = True,
        use_neural_coref: bool = True,
        embedding_provider=None,
        language: str = "en",
    ):
        """Initialize coreference resolver.
        
        Args:
            use_nlp: Whether to use NLP library (spaCy) for resolution. Default True.
            use_neural_coref: Whether to use neural coreference (neuralcoref/coreferee). Default True.
                              Falls back to basic spaCy if neural coreference not available.
            embedding_provider: Optional embedding provider for semantic similarity-based resolution.
                               If provided, will use semantic similarity to disambiguate pronouns in multi-entity contexts.
            language: Language hint for heuristics. Default "en". Use "tr" to enable Turkish pronoun heuristics.
        """
        self.use_nlp = use_nlp
        self.use_neural_coref = use_neural_coref
        self.embedding_provider = embedding_provider
        self.language = (language or "en").lower()
        self.nlp = None
        self.neural_coref = None
        # Conservative safety knobs: prefer abstaining over wrong resolution.
        self.min_resolution_score = 0.85
        self.min_resolution_margin = 0.15
        
        if use_nlp:
            try:
                import spacy
                # Try to load spaCy English model
                try:
                    # Try en_core_web_sm first (smallest, most common)
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model 'en_core_web_sm' loaded for coreference resolution")
                except OSError:
                    try:
                        # Fallback to en_core_web_md
                        self.nlp = spacy.load("en_core_web_md")
                        logger.info("spaCy model 'en_core_web_md' loaded for coreference resolution")
                    except OSError:
                        logger.warning(
                            "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
                        )
                        self.use_nlp = False
                        self.nlp = None
                        return
                
                # Try neural coreference if requested
                if use_neural_coref and self.nlp:
                    # Try coreferee (more modern, actively maintained)
                    try:
                        import coreferee
                        self.nlp.add_pipe('coreferee')
                        self.neural_coref = "coreferee"
                        logger.info("Neural coreference (coreferee) enabled")
                    except (ImportError, Exception) as e:
                        # Try neuralcoref (older, but still works)
                        try:
                            import neuralcoref
                            self.nlp.add_pipe(neuralcoref.NeuralCoref(self.nlp.vocab), name='neuralcoref')
                            self.neural_coref = "neuralcoref"
                            logger.info("Neural coreference (neuralcoref) enabled")
                        except (ImportError, Exception) as e2:
                            logger.warning(
                                f"Neural coreference not available (coreferee: {e}, neuralcoref: {e2}). "
                                "Install with: pip install coreferee or pip install neuralcoref. "
                                "Falling back to basic spaCy coreference."
                            )
                            self.neural_coref = None
            except ImportError:
                logger.warning(
                    "spaCy not available. Install with: pip install spacy "
                    "or pip install drg[coreference]"
                )
                self.use_nlp = False
                self.nlp = None
    
    def resolve(
        self,
        text: str,
        entities: List[Tuple[str, str]],
        relations: List[Tuple[str, str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Resolve pronouns and references in text to explicit entities.
        
        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples
            relations: List of (source, relation, target) tuples
        
        Returns:
            Tuple of (resolved_entities, resolved_relations) where pronouns
            and references are replaced with explicit entity names
        """
        if not entities or not text:
            return entities, relations
        
        if self.use_nlp and self.nlp:
            # Use NLP-based coreference resolution
            return self._resolve_with_nlp(text, entities, relations)
        else:
            # Basic heuristic-based resolution (simple pronoun resolution)
            return self._resolve_with_heuristics(text, entities, relations)
    
    def _resolve_with_nlp(self, text: str, entities: List[Tuple[str, str]], relations: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Resolve coreferences using spaCy NLP with optional neural coreference.
        
        Uses neural coreference (coreferee/neuralcoref) if available, otherwise falls back
        to basic spaCy sentence structure analysis.
        
        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples
            relations: List of (source, relation, target) tuples
        
        Returns:
            Tuple of (resolved_entities, resolved_relations)
        """
        if not self.nlp:
            return entities, relations
        
        try:
            doc = self.nlp(text)
            entity_names = {name for name, _ in entities}
            pronoun_to_entity: Dict[str, str] = {}
            
            # Use neural coreference if available (more accurate)
            if self.neural_coref == "coreferee":
                # coreferee provides coref chains
                if hasattr(doc._, 'coref_chains'):
                    for chain in doc._.coref_chains:
                        # Get the main mention (usually first)
                        main_mention = chain.main
                        main_text = doc[main_mention[0]:main_mention[1]].text
                        
                        # Check if main mention matches an entity
                        if main_text in entity_names:
                            # Map all other mentions in the chain to this entity
                            for mention in chain:
                                mention_text = doc[mention[0]:mention[1]].text
                                if mention_text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
                                    pronoun_to_entity[mention_text] = main_text
            
            elif self.neural_coref == "neuralcoref":
                # neuralcoref provides clusters
                if hasattr(doc._, 'coref_clusters'):
                    for cluster in doc._.coref_clusters:
                        # Get the main mention (first in cluster)
                        main_mention = cluster.main.text
                        
                        # Check if main mention matches an entity
                        if main_mention in entity_names:
                            # Map all mentions in cluster to this entity
                            for mention in cluster.mentions:
                                mention_text = mention.text
                                if mention_text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
                                    pronoun_to_entity[mention_text] = main_mention
            
            # Fallback to improved heuristic if neural coreference didn't work or not available
            # This uses sentence structure and entity type/gender matching for better resolution
            if not pronoun_to_entity:
                sentences = list(doc.sents)
                pronouns = {
                    'he': 'male',
                    'she': 'female',
                    'him': 'male',
                    'her': 'female',
                    'his': 'male',
                    'hers': 'female',
                    'it': 'neutral',
                    'its': 'neutral',
                    'they': 'plural',
                    'them': 'plural',
                    'their': 'plural',
                    'theirs': 'plural'
                }
                
                # Build entity type map for better matching
                entity_type_map = {name: etype for name, etype in entities}
                
                for sent_idx, sent in enumerate(sentences):
                    # Find pronouns in this sentence
                    for token in sent:
                        pronoun_lower = token.text.lower()
                        if pronoun_lower not in pronouns:
                            continue
                        
                        pronoun_gender = pronouns[pronoun_lower]
                        pronoun_pos = token.i
                        best_match = None
                        best_score = 0
                        second_best_score = 0
                        
                        # Strategy 1: Look in current sentence (most common case)
                        # Search backwards from pronoun position
                        pronoun_sentence = next((sent for sent in sentences if token in sent), None)
                        pronoun_context = pronoun_sentence.text.lower() if pronoun_sentence else ""
                        
                        for prev_token in doc[max(0, pronoun_pos - 50):pronoun_pos]:
                            for entity_name in entity_names:
                                entity_words = entity_name.lower().split()
                                score = 0
                                
                                # Check if entity appears in token
                                if any(entity_word in prev_token.text.lower() or 
                                       entity_word in prev_token.lemma_.lower() 
                                       for entity_word in entity_words):
                                    score = 1.0
                                    # Boost score if entity type matches expected gender
                                    # (Person entities are more likely for he/she pronouns)
                                    entity_type = entity_type_map.get(entity_name, '')
                                    if pronoun_gender in ['male', 'female'] and entity_type == 'Person':
                                        score = 1.5
                                    # Boost if close to pronoun
                                    distance = pronoun_pos - prev_token.i
                                    score *= (1.0 / (1.0 + distance * 0.1))
                                    
                                    # Strategy 1a: Semantic similarity scoring (if embedding provider available)
                                    # Use semantic similarity to disambiguate in multi-entity contexts
                                    if self.embedding_provider and pronoun_context:
                                        semantic_score = self._get_semantic_similarity_score(
                                            entity_name, pronoun_context, pronoun_pos, doc
                                        )
                                        if semantic_score > 0:
                                            score *= (1.0 + semantic_score * 0.3)  # Boost up to 30% based on semantic similarity
                                    
                                    # Strategy 1b: Action-based matching
                                    # If pronoun appears in action context, prefer entities related to that action
                                    action_score = self._get_action_based_score(
                                        entity_name, pronoun_context, entity_type_map.get(entity_name, '')
                                    )
                                    if action_score > 0:
                                        score *= (1.0 + action_score * 0.2)  # Boost up to 20% based on action context
                                    
                                    if score > best_score:
                                        second_best_score = best_score
                                        best_match = entity_name
                                        best_score = score
                                    elif score > second_best_score:
                                        second_best_score = score
                        
                        # Strategy 2: If no match, check previous sentences (up to 2 sentences back)
                        if not best_match or best_score < 0.8:
                            for prev_sent_idx in range(max(0, sent_idx - 2), sent_idx):
                                prev_sent = sentences[prev_sent_idx]
                                prev_sent_text = prev_sent.text.lower()
                                for entity_name in entity_names:
                                    entity_words = entity_name.lower().split()
                                    # Check if entity appears in previous sentence
                                    if any(word in prev_sent_text for word in entity_words):
                                        score = 0.7  # Lower score for previous sentences
                                        entity_type = entity_type_map.get(entity_name, '')
                                        if pronoun_gender in ['male', 'female'] and entity_type == 'Person':
                                            score = 1.0
                                        
                                        # Strategy 2a: Semantic similarity for previous sentence context
                                        if self.embedding_provider:
                                            combined_context = f"{prev_sent_text} {pronoun_context}"
                                            semantic_score = self._get_semantic_similarity_score(
                                                entity_name, combined_context, pronoun_pos, doc
                                            )
                                            if semantic_score > 0:
                                                score *= (1.0 + semantic_score * 0.25)
                                        
                                        # Strategy 2b: Subject-verb-object pattern matching
                                        # If pronoun is in object position and previous sentence has subject-verb pattern
                                        if self._matches_svo_pattern(entity_name, prev_sent, pronoun_context):
                                            score *= 1.3
                                        
                                        if score > best_score:
                                            second_best_score = best_score
                                            best_match = entity_name
                                            best_score = score
                                        elif score > second_best_score:
                                            second_best_score = score
                        
                        # Strategy 3 (conservative): if pronoun implies Person and there is exactly one Person entity,
                        # resolve to it. Otherwise, abstain.
                        if not best_match and pronoun_gender in {"male", "female"}:
                            persons = [e for e in entity_names if entity_type_map.get(e, "") == "Person"]
                            if len(persons) == 1:
                                best_match = persons[0]
                                best_score = max(best_score, 0.90)
                                second_best_score = max(second_best_score, 0.0)
                        
                        # Map pronoun to best matching entity
                        if best_match:
                            # Conservative gating: only map if high confidence and unambiguous.
                            if (
                                best_score >= self.min_resolution_score
                                and (second_best_score <= 0 or (best_score - second_best_score) >= self.min_resolution_margin)
                            ):
                                pronoun_to_entity[token.text] = best_match
                            else:
                                logger.debug(
                                    f"Pronoun '{token.text}' resolution ambiguous/low "
                                    f"(best={best_score:.2f}, second={second_best_score:.2f}), skipping"
                                )
            
            # Replace pronouns in relations
            if pronoun_to_entity:
                resolved_relations = []
                for s, r, o in relations:
                    new_s = pronoun_to_entity.get(s, s)
                    new_o = pronoun_to_entity.get(o, o)
                    resolved_relations.append((new_s, r, new_o))
                
                coref_type = self.neural_coref or "basic"
                logger.info(f"Coreference resolution ({coref_type}): {len(pronoun_to_entity)} pronouns resolved")
                return entities, resolved_relations
            else:
                logger.debug("Coreference resolution: No pronouns found or resolved")
                return entities, relations
                
        except Exception as e:
            logger.warning(f"NLP-based coreference resolution failed: {e}, falling back to heuristics")
            return self._resolve_with_heuristics(text, entities, relations)
    
    def _resolve_with_heuristics(self, text: str, entities: List[Tuple[str, str]], relations: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Resolve coreferences using improved heuristics (sentence structure + entity type matching).
        
        This is a fallback implementation when NLP is not available. It uses:
        - Sentence structure analysis (pronouns likely refer to recent entities)
        - Entity type matching (he/she → Person, it → non-Person)
        - Distance-based scoring (closer entities have higher priority)
        - Multiple sentence context (check up to 2 sentences back)
        
        **Note**: This is still inferior to neural coreference resolution. For best results,
        install coreferee: pip install coreferee
        
        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples
            relations: List of (source, relation, target) tuples
        
        Returns:
            Tuple of (resolved_entities, resolved_relations)
        """
        if not entities or not relations:
            return entities, relations
        
        sentences = re.split(r'[.!?]+\s+', text)
        entity_names = {name for name, _ in entities}
        entity_type_map = {name: etype for name, etype in entities}
        pronoun_to_entity: Dict[str, str] = {}
        
        pronouns = {
            'he': 'male',
            'she': 'female',
            'him': 'male',
            'her': 'female',
            'his': 'male',
            'hers': 'female',
            'it': 'neutral',
            'its': 'neutral',
            'they': 'plural',
            'them': 'plural',
            'their': 'plural',
            'theirs': 'plural',
        }
        if self.language in {"tr", "turkish"}:
            # Turkish (gender-neutral)
            pronouns.update(
                {
                    'o': 'ambiguous',
                    'ona': 'ambiguous',
                    'onu': 'ambiguous',
                    'onun': 'ambiguous',
                    'onlar': 'plural',
                    'onlara': 'plural',
                    'onları': 'plural',
                    'onların': 'plural',
        }
            )
        
        # Map pronouns in relations to entities using improved heuristics
        for s, r, o in relations:
            # Check source
            if s.lower() in pronouns:
                best_match = self._find_entity_for_pronoun(
                    s, text, sentences, entity_names, entity_type_map, pronouns[s.lower()]
                )
                if best_match:
                    pronoun_to_entity[s] = best_match
            
            # Check object
            if o.lower() in pronouns:
                best_match = self._find_entity_for_pronoun(
                    o, text, sentences, entity_names, entity_type_map, pronouns[o.lower()]
                )
                if best_match:
                    pronoun_to_entity[o] = best_match
        
        # Replace pronouns in relations
        if pronoun_to_entity:
            resolved_relations = []
            for s, r, o in relations:
                new_s = pronoun_to_entity.get(s, s)
                new_o = pronoun_to_entity.get(o, o)
                resolved_relations.append((new_s, r, new_o))
            
            logger.info(f"Coreference resolution (improved heuristics): {len(pronoun_to_entity)} pronouns resolved")
            return entities, resolved_relations
        
        logger.debug("Coreference resolution (heuristics): No pronouns resolved")
        return entities, relations
    
    def _find_entity_for_pronoun(
        self,
        pronoun: str,
        text: str,
        sentences: List[str],
        entity_names: Set[str],
        entity_type_map: Dict[str, str],
        pronoun_gender: str
    ) -> Optional[str]:
        """Find the best matching entity for a pronoun using heuristics.
        
        Args:
            pronoun: The pronoun text (e.g., "he", "she", "it")
            text: Full text
            sentences: List of sentences
            entity_names: Set of entity names
            entity_type_map: Map of entity name to entity type
            pronoun_gender: Gender/type hint ('male', 'female', 'neutral', 'plural')
        
        Returns:
            Best matching entity name or None
        """
        pronoun_lower = pronoun.lower()
        # Find a whole-word pronoun occurrence to avoid substring collisions (e.g., "o" in "Elon").
        m = re.search(rf"(?i)(?<!\w){re.escape(pronoun_lower)}(?!\w)", text)
        if not m:
            return None
        pronoun_index = m.start()
        
        # Determine sentence boundaries in the ORIGINAL text (do not rely on pre-split sentences,
        # which may not preserve offsets/punctuation).
        def _sent_bounds_around(idx: int) -> Tuple[int, int]:
            left = max(text.rfind(".", 0, idx), text.rfind("?", 0, idx), text.rfind("!", 0, idx))
            start = 0 if left == -1 else left + 1
            # Find next delimiter after idx
            right_candidates = [p for p in (text.find(".", idx), text.find("?", idx), text.find("!", idx)) if p != -1]
            end = (min(right_candidates) + 1) if right_candidates else len(text)
            return start, end

        cur_start, cur_end = _sent_bounds_around(pronoun_index)
        current_sent = text[cur_start:cur_end].strip().lower()
        # Previous sentence windows (up to 2)
        prev_sents: List[str] = []
        prev_end = cur_start - 1
        for _ in range(2):
            if prev_end <= 0:
                break
            left = max(text.rfind(".", 0, prev_end), text.rfind("?", 0, prev_end), text.rfind("!", 0, prev_end))
            start = 0 if left == -1 else left + 1
            prev_sents.append(text[start:prev_end + 1].strip().lower())
            prev_end = start - 2
        
        best_match = None
        best_score = 0.0
        second_best_score = 0.0
        
        # Strategy 1: Look in current sentence (most common case)
        if current_sent:
            sent_tokens = re.findall(r"\b\w+\b", current_sent)
            for entity_name in entity_names:
                entity_words = entity_name.lower().split()
                score = 0
                
                # Check if entity appears in current sentence before pronoun
                entity_mentions = [
                    i for i, word in enumerate(sent_tokens)
                    if any(ew in word for ew in entity_words)
                ]
                pronoun_pos_in_sent = (
                    sent_tokens.index(pronoun_lower) if pronoun_lower in sent_tokens else -1
                )
                
                if entity_mentions and pronoun_pos_in_sent > 0:
                    # Entity mentioned before pronoun in same sentence
                    before_pronoun = [pos for pos in entity_mentions if pos < pronoun_pos_in_sent]
                    if before_pronoun:
                        score = 1.0
                        # Boost score based on distance (closer = better)
                        distance = pronoun_pos_in_sent - max(before_pronoun)
                        score *= (1.0 / (1.0 + distance * 0.2))
                        # Boost if entity type matches pronoun type
                        entity_type = entity_type_map.get(entity_name, '')
                        if pronoun_gender in ['male', 'female'] and entity_type == 'Person':
                            score *= 1.5
                        elif pronoun_gender == 'neutral' and entity_type != 'Person':
                            score *= 1.5
                        
                        if score > best_score:
                            second_best_score = best_score
                            best_match = entity_name
                            best_score = score
                        elif score > second_best_score:
                            second_best_score = score
        
        # Strategy 2: Check previous sentences (up to 2 sentences back), but abstain if ambiguous.
        if not best_match or best_score < 0.7:
            for prev_sent in prev_sents:
                if not prev_sent:
                    continue
                # If multiple Person candidates exist in the same previous sentence and pronoun is person-like,
                # abstain to avoid guessing (input-agnostic safety).
                if pronoun_gender in {"male", "female", "ambiguous"}:
                    person_hits = [
                        e for e in entity_names
                        if entity_type_map.get(e, "") == "Person" and e.lower() in prev_sent
                    ]
                    if len(person_hits) >= 2:
                        continue

                # Rank entities by first mention position (salience proxy).
                positions: List[Tuple[int, str]] = []
                for entity_name in entity_names:
                    pos = prev_sent.find(entity_name.lower())
                    if pos != -1:
                        positions.append((pos, entity_name))
                positions.sort(key=lambda x: x[0])
                if not positions:
                    continue

                for rank, (_, entity_name) in enumerate(positions):
                    # Base score: first mention is often the subject/salient entity.
                    score = 0.9 if rank == 0 else 0.7
                    entity_type = entity_type_map.get(entity_name, "")
                    if pronoun_gender in {"male", "female"} and entity_type == "Person":
                        score *= 1.1
                    if pronoun_gender == "neutral" and entity_type and entity_type != "Person":
                        score *= 1.1
                    if score > best_score:
                        second_best_score = best_score
                        best_match = entity_name
                        best_score = score
                    elif score > second_best_score:
                        second_best_score = score
        
        # Conservative fallback: if pronoun implies Person and there is exactly one Person entity, pick it.
        if not best_match and pronoun_gender in {"male", "female"}:
            persons = [e for e in entity_names if entity_type_map.get(e, "") == "Person"]
            if len(persons) == 1:
                best_match = persons[0]
                best_score = max(best_score, 0.90)
        
        # Only return if confidence is high enough and unambiguous.
        if best_match and (
            best_score >= self.min_resolution_score
            and (second_best_score <= 0 or (best_score - second_best_score) >= self.min_resolution_margin)
        ):
            return best_match
        return None
    
    def _get_semantic_similarity_score(
        self, entity_name: str, context: str, pronoun_pos: int, doc
    ) -> float:
        """Calculate semantic similarity score between entity and pronoun context.
        
        Uses embedding provider if available to disambiguate pronouns in multi-entity contexts.
        For example, "iPhone hakkında konuştu" context → Tim Cook (more semantically related than Elon Musk).
        
        Args:
            entity_name: Entity name to check
            context: Text context around the pronoun
            pronoun_pos: Position of pronoun in doc
            doc: spaCy doc object
        
        Returns:
            Similarity score between 0 and 1, or 0 if embeddings not available
        """
        if not self.embedding_provider:
            return 0.0
        
        try:
            # Extract meaningful context around pronoun (avoid just pronouns)
            context_window = doc[max(0, pronoun_pos - 10):pronoun_pos + 10].text
            if not context_window.strip():
                return 0.0
            
            # Get embeddings
            entity_emb = self.embedding_provider.embed(entity_name)
            context_emb = self.embedding_provider.embed(context_window)
            
            # Calculate cosine similarity
            try:
                import numpy as np
            except ImportError:
                # Fallback to basic math if numpy not available
                logger.debug("numpy not available, using basic cosine similarity")
                dot_product = sum(a * b for a, b in zip(entity_emb, context_emb))
                norm_a = sum(a * a for a in entity_emb) ** 0.5
                norm_b = sum(b * b for b in context_emb) ** 0.5
                norm_product = norm_a * norm_b
                if norm_product == 0:
                    return 0.0
                similarity = dot_product / norm_product
                return max(0.0, similarity)
            
            dot_product = np.dot(entity_emb, context_emb)
            norm_product = np.linalg.norm(entity_emb) * np.linalg.norm(context_emb)
            if norm_product == 0:
                return 0.0
            
            similarity = dot_product / norm_product
            # Normalize to 0-1 range (cosine similarity is already -1 to 1, but typically 0-1 for embeddings)
            return max(0.0, similarity)
        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _get_action_based_score(self, entity_name: str, context: str, entity_type: str) -> float:
        """Calculate action-based matching score.
        
        If pronoun appears in action context (e.g., "spoke about iPhone"), prefer entities
        semantically related to that action context.
        
        Args:
            entity_name: Entity name to check
            context: Text context around the pronoun
            entity_type: Type of entity
        
        Returns:
            Action-based score between 0 and 1
        """
        # Action keywords that indicate topic/subject focus
        action_keywords = {
            'spoke': ['about', 'regarding', 'on', 'concerning'],
            'discussed': ['about', 'regarding'],
            'mentioned': ['about'],
            'talked': ['about', 'on'],
            'wrote': ['about', 'on'],
            'created': ['by'],
            'designed': ['by'],
            'developed': ['by'],
        }
        
        context_lower = context.lower()
        score = 0.0
        
        # Check if context contains action keywords
        for action, prepositions in action_keywords.items():
            if action in context_lower:
                # If action has a preposition, the following phrase is likely the topic
                for prep in prepositions:
                    pattern = f"{action}.*{prep}"
                    if re.search(pattern, context_lower):
                        # Entity name in same context suggests relevance
                        if entity_name.lower() in context_lower:
                            score = 0.5
                        # For Person entities with topic-related actions, boost score
                        if entity_type == 'Person':
                            score = 0.3  # Person entities are relevant in action contexts
        
        return score
    
    def _matches_svo_pattern(self, entity_name: str, prev_sent, pronoun_context: str) -> bool:
        """Check if entity matches subject-verb-object pattern from previous sentence.
        
        If previous sentence has "EntityA verb Object", and pronoun context mentions the object,
        EntityA is more likely to be the referent.
        
        Args:
            entity_name: Entity name to check
            prev_sent: Previous sentence (spaCy Span)
            pronoun_context: Context around pronoun
        
        Returns:
            True if matches SVO pattern
        """
        try:
            # Simple pattern: if entity is subject of previous sentence and pronoun context mentions object
            prev_sent_lower = prev_sent.text.lower()
            pronoun_context_lower = pronoun_context.lower()
            entity_lower = entity_name.lower()
            
            # Check if entity appears as subject in previous sentence
            entity_is_subject = False
            for token in prev_sent:
                if token.text.lower() == entity_lower and token.dep_ in ['nsubj', 'nsubjpass']:
                    entity_is_subject = True
                    break
            
            if entity_is_subject:
                # Check if pronoun context mentions object-related terms
                # This is a heuristic - in practice, neural coreference handles this better
                object_keywords = ['about', 'regarding', 'concerning', 'on', 'for', 'with']
                if any(keyword in pronoun_context_lower for keyword in object_keywords):
                    return True
        
        except Exception:
            pass
        
        return False


def resolve_coreferences(
    text: str,
    entities: List[Tuple[str, str]],
    relations: List[Tuple[str, str, str]],
    use_nlp: bool = True,
    use_neural_coref: bool = True,
    embedding_provider=None,
    language: str = "en",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Convenience function to resolve coreferences.
    
    Resolves pronouns and generic references to explicit entities using multiple strategies:
    1. Neural coreference resolution (coreferee/neuralcoref) if available
    2. Semantic similarity-based disambiguation (if embedding_provider provided)
    3. Action-based context matching
    4. Subject-verb-object pattern matching
    5. Improved heuristics (entity type matching, distance-based scoring)
    
    This function works for ANY domain - technology, business, science, medicine, etc.
    The resolution strategies are domain-agnostic and work with any entity types.
    
    Args:
        text: Original text
        entities: List of (entity_name, entity_type) tuples
        relations: List of (source, relation, target) tuples
        use_nlp: Whether to use NLP library (spaCy) for resolution. Default True.
                  Falls back to heuristics if NLP not available.
        use_neural_coref: Whether to use neural coreference (coreferee/neuralcoref). Default True.
                          Falls back to basic spaCy if neural coreference not available.
        embedding_provider: Optional embedding provider for semantic similarity-based disambiguation.
                           If provided, uses semantic similarity to disambiguate pronouns in multi-entity contexts.
                           Example: "EntityA and EntityB met. He spoke about Topic." 
                           → Semantic similarity between Topic and EntityA/EntityB helps disambiguate "He".
    
    Returns:
        Tuple of (resolved_entities, resolved_relations) where pronouns and generic references
        are replaced with explicit entity names
    
    Example:
        text = "EntityA and EntityB met. He spoke about Topic."
        entities = [("EntityA", "Person"), ("EntityB", "Person"), ("Topic", "Concept")]
        relations = [("He", "spoke_about", "Topic")]
        
        # Without embedding: May incorrectly resolve "He" to EntityA (first Person)
        # With embedding: Semantic similarity (Topic ↔ EntityB) → correctly resolves "He" to EntityB
    """
    resolver = CoreferenceResolver(
        use_nlp=use_nlp, 
        use_neural_coref=use_neural_coref,
        embedding_provider=embedding_provider,
        language=language,
    )
    return resolver.resolve(text, entities, relations)

