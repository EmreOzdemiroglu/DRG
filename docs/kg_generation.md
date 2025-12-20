# Knowledge Graph Generation

## Overview

The Knowledge Graph Core module provides a comprehensive, modular monolith architecture for representing and managing knowledge graphs. It is designed to be dataset-agnostic and supports multiple export formats compatible with GraphRAG and other semantic web standards.

## Architecture

### Modular Monolith Design

The KG core follows a modular monolith pattern where all components are in the same codebase but have clear boundaries:

- **Entities Module**: Node representation with type, properties, and metadata
- **Relationships Module**: Edge representation with enriched relationship details
- **Clusters Module**: Community/cluster representation (algorithm-agnostic)
- **Visualization Module**: Export to various visualization formats
- **Export Module**: Multiple format exports (JSON, JSON-LD, GraphRAG)

### Core Components

#### KGNode
Represents entities in the knowledge graph:
- **id**: Unique identifier
- **type**: Entity type (e.g., "Person", "Location", "Event")
- **properties**: Flexible dictionary of entity properties
- **metadata**: Confidence scores, source references, etc.

#### KGEdge
Represents relationships between entities:
- **source**: Source entity identifier
- **target**: Target entity identifier
- **relationship_type**: Type from taxonomy
- **relationship_detail**: Natural language explanation
- **metadata**: Confidence, source references, etc.

#### Cluster
Represents communities/clusters of related entities:
- **id**: Cluster identifier
- **node_ids**: Set of node identifiers in cluster
- **metadata**: Additional cluster information

#### EnhancedKG
Main knowledge graph container:
- Manages nodes, edges, and clusters
- Provides export functionality
- Supports graph operations

## Export Formats

### JSON Format
Standard JSON representation suitable for:
- API responses
- Configuration storage
- Inter-system communication

Structure:
```json
{
  "nodes": [...],
  "edges": [...],
  "clusters": [...]
}
```

### JSON-LD Format
Linked data format compatible with semantic web standards:
- Uses @context for vocabulary
- Supports semantic web tools
- Enables RDF conversion

### GraphRAG Format
Microsoft GraphRAG-compatible format:
- Uses "entities" instead of "nodes"
- Includes "communities" for clusters
- Optimized for GraphRAG processing pipelines

## Metadata Support

### Node Metadata
- **confidence**: Extraction confidence score
- **source_ref**: Reference to source document/chunk
- Custom metadata fields as needed

### Edge Metadata
- **confidence**: Relationship confidence score
- **source_ref**: Reference to source text
- Relationship-specific metadata

## Design Principles

### Dataset-Agnostic
- No hardcoded domain assumptions
- Flexible property system
- Extensible entity types

### Algorithm-Agnostic Clustering
- Clusters accepted as input
- No clustering algorithm implementation
- Works with any clustering method (Louvain, Leiden, Spectral, etc.)

### Declarative Over Imperative
- Schema-driven structure
- Configuration over code
- Export formats defined declaratively

## Use Cases

### Long-Text Analysis
- Supports large document collections
- Efficient metadata tracking
- Scalable representation

### Multi-Dataset Comparison
- Consistent format across datasets
- Metadata enables comparison
- Standardized exports

### Explainable Graphs
- Relationship details provide explanations
- Confidence scores indicate reliability
- Source references enable traceability

## Trade-offs

### Advantages
- Single codebase (monolith) but clear modules
- Multiple export formats from one representation
- Flexible metadata system
- Algorithm-agnostic design

### Limitations
- In-memory representation (may need persistence layer for large graphs)
- No built-in graph algorithms (focus on representation)
- Clustering must be done externally

## Future Enhancements

Potential improvements:
- Graph persistence layer
- Incremental graph building
- Graph versioning
- Query interface
- Graph transformation operations





