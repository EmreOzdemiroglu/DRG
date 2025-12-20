# Relationship Model - Enriched Relationships

## Overview

The relationship modeling system provides enriched relationship representation compatible with GraphRAG format. It includes a comprehensive taxonomy of relationship types, classification mechanisms, and structured relationship details.

## Relationship Structure

### EnrichedRelationship Format

Each relationship follows this structure:
- **source**: Source entity identifier
- **target**: Target entity identifier
- **relationship_type**: Type from taxonomy
- **relationship_detail**: Short natural language explanation
- **confidence**: Confidence score (0.0 to 1.0)
- **source_ref**: Reference to source (e.g., chunk_id, document_id)

### Relationship Detail

The relationship_detail field contains a short, natural language explanation of the relationship. This provides:
- **Explainability**: Why this relationship exists
- **Context**: What the relationship means in the specific domain
- **Traceability**: Human-readable justification

Example: "Alice influences Bob's decision-making process through strategic advice."

## Relationship Type Taxonomy

### Causal Relationships
- **causes**: A directly causes B
- **caused_by**: A is caused by B
- **triggers**: A triggers B
- **results_in**: A results in B

### Spatial Relationships
- **located_at**: A is located at B
- **contains**: A contains B
- **near**: A is near B
- **inside/outside**: Spatial containment

### Temporal Relationships
- **occurs_before**: A occurs before B
- **occurs_after**: A occurs after B
- **occurs_during**: A occurs during B
- **follows**: A follows B

### Social/Interaction Relationships
- **influences**: A influences B
- **influenced_by**: A is influenced by B
- **collaborates_with**: A collaborates with B
- **works_with**: A works with B
- **owns**: A owns B
- **belongs_to**: A belongs to B
- **member_of**: A is a member of B

### Hierarchical Relationships
- **parent_of**: A is parent of B
- **child_of**: A is child of B
- **part_of**: A is part of B
- **has_part**: A has part B

### Action Relationships
- **creates**: A creates B
- **destroys**: A destroys B
- **modifies**: A modifies B
- **produces**: A produces B
- **consumes**: A consumes B

### Communication Relationships
- **communicates_with**: A communicates with B
- **informs**: A informs B
- **requests**: A requests from B
- **responds_to**: A responds to B

### Emotional/Subjective Relationships
- **likes/dislikes**: A likes/dislikes B
- **loves/hates**: Strong emotional connection
- **fears**: A fears B
- **trusts**: A trusts B

And many more domain-agnostic types.

## Classification System

### Hybrid Classifier Design

The relationship classification system uses a hybrid approach:

#### Rule-Based Classification
- Pattern matching on text
- Type compatibility heuristics
- Schema constraint checking
- Fast and deterministic

#### LLM-Based Classification (Stub)
- Placeholder for future implementation
- Will use LLM for complex cases
- Explanation generation
- Higher accuracy for ambiguous cases

### Classification Process

1. **Pattern Matching**: Match raw text against patterns
2. **Type Heuristics**: Use entity types to infer likely relationships
3. **Schema Constraints**: Filter by schema-defined valid relationships
4. **LLM Classification**: (Future) Use LLM for difficult cases
5. **Confidence Scoring**: Assign confidence based on method used

## Dataset Independence

The relationship model is designed to work across domains:

- **Taxonomy**: General-purpose relationship types
- **No Domain Assumptions**: Works with narrative, factual, technical text
- **Extensible**: New relationship types can be added
- **Flexible Details**: Relationship_detail adapts to domain

## Use Cases

### Narrative Text
- Character relationships (influences, collaborates_with)
- Plot relationships (caused_by, triggers)
- Spatial relationships (located_at, visits)

### Factual Text
- Historical relationships (influences, member_of)
- Causal relationships (causes, results_in)
- Temporal relationships (occurs_before, occurs_during)

### Technical Documents
- System relationships (contains, part_of)
- Process relationships (produces, consumes)
- Dependency relationships (depends_on, uses)

## Trade-offs

### Advantages
- Rich taxonomy covers many relationship types
- Explainable relationships through detail field
- Hybrid classification balances speed and accuracy
- Dataset-agnostic design

### Limitations
- Taxonomy may not cover all domain-specific relationships
- Rule-based classification has limited accuracy
- LLM-based classification not yet implemented
- Relationship_detail generation requires external logic

## Future Enhancements

Potential improvements:
- Implement LLM-based classification
- Automatic relationship_detail generation
- Domain-specific taxonomy extensions
- Relationship confidence calibration
- Relationship validation rules





