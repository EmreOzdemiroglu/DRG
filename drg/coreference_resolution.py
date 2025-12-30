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
    
    def __init__(self, use_nlp: bool = True):
        """Initialize coreference resolver.
        
        Args:
            use_nlp: Whether to use NLP library (spaCy) for resolution. Default True.
        """
        self.use_nlp = use_nlp
        self.nlp = None
        
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
        """Resolve coreferences using spaCy NLP.
        
        Uses sentence structure and entity mentions to resolve pronouns to entities.
        
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
            
            # Build pronoun-to-entity mapping using sentence structure
            pronoun_to_entity: Dict[str, str] = {}
            
            # Strategy: Find pronouns and link them to the most recent matching entity in the sentence/previous sentence
            sentences = list(doc.sents)
            pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs']
            
            for sent_idx, sent in enumerate(sentences):
                sent_text = sent.text.lower()
                sent_tokens = [token.text.lower() for token in sent]
                
                # Find pronouns in this sentence
                for token in sent:
                    if token.text.lower() in pronouns:
                        # Find the most recent entity mentioned before this pronoun
                        # Look in current sentence first, then previous sentences
                        best_match = None
                        best_distance = float('inf')
                        
                        # Search current sentence (backwards from pronoun)
                        pronoun_pos = token.i
                        for prev_token in doc[pronoun_pos - 20:pronoun_pos]:  # Look back up to 20 tokens
                            prev_text = prev_token.text
                            # Check if this token or its lemma matches an entity
                            for entity_name in entity_names:
                                entity_words = entity_name.lower().split()
                                # Check if entity name appears in nearby tokens
                                if any(entity_word in prev_token.text.lower() or 
                                       entity_word in prev_token.lemma_.lower() 
                                       for entity_word in entity_words):
                                    distance = pronoun_pos - prev_token.i
                                    if distance < best_distance:
                                        best_match = entity_name
                                        best_distance = distance
                                        break
                        
                        # If no match in current sentence, check previous sentence
                        if not best_match and sent_idx > 0:
                            prev_sent = sentences[sent_idx - 1]
                            for entity_name in entity_names:
                                entity_words = entity_name.lower().split()
                                if any(word in prev_sent.text.lower() for word in entity_words):
                                    best_match = entity_name
                                    break
                        
                        # If found a match, map pronoun to entity
                        if best_match:
                            pronoun_lower = token.text.lower()
                            # Map various forms of the pronoun
                            if pronoun_lower in ['he', 'him', 'his']:
                                pronoun_to_entity[token.text] = best_match
                                # Also map variations
                                for variant in ['he', 'him', 'his']:
                                    if variant in sent_text and variant not in pronoun_to_entity:
                                        pronoun_to_entity[variant] = best_match
                            elif pronoun_lower in ['she', 'her', 'hers']:
                                pronoun_to_entity[token.text] = best_match
                                for variant in ['she', 'her', 'hers']:
                                    if variant in sent_text and variant not in pronoun_to_entity:
                                        pronoun_to_entity[variant] = best_match
                            elif pronoun_lower in ['it', 'its']:
                                pronoun_to_entity[token.text] = best_match
                                for variant in ['it', 'its']:
                                    if variant in sent_text and variant not in pronoun_to_entity:
                                        pronoun_to_entity[variant] = best_match
                            elif pronoun_lower in ['they', 'them', 'their', 'theirs']:
                                pronoun_to_entity[token.text] = best_match
                                for variant in ['they', 'them', 'their', 'theirs']:
                                    if variant in sent_text and variant not in pronoun_to_entity:
                                        pronoun_to_entity[variant] = best_match
            
            # Replace pronouns in relations
            if pronoun_to_entity:
                resolved_relations = []
                for s, r, o in relations:
                    new_s = pronoun_to_entity.get(s, s)
                    new_o = pronoun_to_entity.get(o, o)
                    resolved_relations.append((new_s, r, new_o))
                
                logger.info(f"Coreference resolution (NLP): {len(pronoun_to_entity)} pronouns resolved")
                return entities, resolved_relations
            else:
                logger.debug("Coreference resolution (NLP): No pronouns found or resolved")
                return entities, relations
                
        except Exception as e:
            logger.warning(f"NLP-based coreference resolution failed: {e}, falling back to heuristics")
            return self._resolve_with_heuristics(text, entities, relations)
    
    def _resolve_with_heuristics(self, text: str, entities: List[Tuple[str, str]], relations: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Resolve coreferences using simple heuristics (sentence structure).
        
        This is a basic implementation that uses simple rules:
        - Pronouns in a sentence likely refer to entities mentioned earlier
        - Most recent entity of matching type is preferred
        
        Args:
            text: Original text
            entities: List of (entity_name, entity_type) tuples
            relations: List of (source, relation, target) tuples
        
        Returns:
            Tuple of (resolved_entities, resolved_relations)
        """
        if not entities or not relations:
            return entities, relations
        
        # Simple heuristic: Find pronouns in relations and try to match them to entities
        # This is very basic - NLP-based resolution is much better
        sentences = re.split(r'[.!?]+', text)
        entity_names = {name for name, _ in entities}
        pronoun_to_entity: Dict[str, str] = {}
        
        # Map pronouns in relations to entities mentioned in the same sentence/context
        for s, r, o in relations:
            # Check if source or object is a pronoun
            if s.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
                # Try to find entity in same context
                for entity_name in entity_names:
                    # Simple check: if entity appears in text before pronoun, map it
                    # This is very basic - NLP would do much better
                    if entity_name.lower() in text.lower():
                        # Check if entity appears before this pronoun context
                        pronoun_index = text.lower().find(s.lower())
                        entity_index = text.lower().find(entity_name.lower())
                        if entity_index < pronoun_index:
                            pronoun_to_entity[s] = entity_name
                            break
            
            if o.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
                for entity_name in entity_names:
                    if entity_name.lower() in text.lower():
                        pronoun_index = text.lower().find(o.lower())
                        entity_index = text.lower().find(entity_name.lower())
                        if entity_index < pronoun_index:
                            pronoun_to_entity[o] = entity_name
                            break
        
        # Replace pronouns in relations
        if pronoun_to_entity:
            resolved_relations = []
            for s, r, o in relations:
                new_s = pronoun_to_entity.get(s, s)
                new_o = pronoun_to_entity.get(o, o)
                resolved_relations.append((new_s, r, new_o))
            
            logger.info(f"Coreference resolution (heuristics): {len(pronoun_to_entity)} pronouns resolved")
            return entities, resolved_relations
        
        logger.debug("Coreference resolution (heuristics): No pronouns resolved")
        return entities, relations


def resolve_coreferences(
    text: str,
    entities: List[Tuple[str, str]],
    relations: List[Tuple[str, str, str]],
    use_nlp: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """Convenience function to resolve coreferences.
    
    Args:
        text: Original text
        entities: List of (entity_name, entity_type) tuples
        relations: List of (source, relation, target) tuples
        use_nlp: Whether to use NLP library (spaCy) for resolution. Default True.
                  Falls back to heuristics if NLP not available.
    
    Returns:
        Tuple of (resolved_entities, resolved_relations)
    """
    resolver = CoreferenceResolver(use_nlp=use_nlp)
    return resolver.resolve(text, entities, relations)

