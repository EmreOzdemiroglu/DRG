# Schema Design - Dataset-Agnostic Entity Schema Generation

## Overview

The schema generation system provides a dataset-agnostic approach to defining entity classes with structured properties. This system is designed to work across different domains and text types without requiring domain-specific modifications to the core schema structure.

## Design Principles

### Dataset-Agnostic Architecture

The schema generation system follows these principles:

1. **Generalization over Specialization**: Entity classes are designed to capture common patterns across domains rather than domain-specific details.

2. **Property-Based Flexibility**: Instead of hard-coding domain-specific attributes, the system uses a flexible property system where each property has:
   - `name`: Unique identifier for the property
   - `description`: Human-readable explanation
   - `example_value`: Concrete example showing expected data type and format

3. **Extensibility**: While default entity classes (Person, Location, Event) are provided, the system allows easy addition of custom entity classes without modifying core code.

## Entity Class Structure

### Default Entity Classes

#### Person
Represents human individuals with psychological and relational attributes:
- **emotion**: Current emotional state
- **intent**: Primary goals or intentions
- **traits**: Character traits (list)
- **relationships**: Relationships with others (dict)
- **role**: Position or role in context
- **age**: Age or age range

#### Location
Represents places with spatial and symbolic meaning:
- **atmosphere**: Emotional/sensory atmosphere
- **symbolism**: Symbolic representation
- **type**: Category of location
- **coordinates**: Geographical coordinates (optional)
- **features**: Notable characteristics (list)

#### Event
Represents occurrences with temporal and causal aspects:
- **actors**: Entities involved (list)
- **outcomes**: Results or consequences
- **temporal_scope**: Time period or duration
- **type**: Category of event
- **significance**: Importance level
- **cause**: Trigger or cause

## Property Definition System

Each property in an entity class follows a structured format:

```
PropertyDefinition(
    name: str,
    description: str,
    example_value: Any
)
```

The `example_value` can be of any Python type (str, int, list, dict, etc.), allowing flexibility in data representation.

## Schema Export Formats

### JSON Format
Standard JSON representation suitable for:
- Configuration storage
- API responses
- Inter-system communication

### YAML Format
Human-readable YAML format suitable for:
- Manual editing
- Documentation
- Configuration files

## Extension Mechanism

### Adding Custom Entity Classes

New entity classes can be added through the `add_entity_class()` method:

1. Create an `EntityClassDefinition` instance
2. Add properties using `add_property()`
3. Register with the schema generator

### Modifying Existing Classes

Existing entity classes can be updated through:
- `update_entity_class()`: Replace entire class definition
- Direct property manipulation (with validation)

## Use Cases

### Narrative Text
- Person entities capture character development
- Location entities capture setting and atmosphere
- Event entities capture plot points

### Factual Text
- Person entities represent historical figures
- Location entities represent geographical entities
- Event entities represent historical occurrences

### Technical Documents
- Can be extended with technical entity classes
- Properties adapted for technical concepts

## Trade-offs

### Advantages
- Works across multiple domains without modification
- Flexible property system accommodates various data types
- Easy to extend with new entity classes

### Limitations
- May require custom properties for domain-specific concepts
- Example values are static and don't capture all possible variations
- Property validation is basic (type checking at runtime)

## Future Enhancements

Potential improvements:
- Property type validation (schema-level type constraints)
- Property relationships (dependencies between properties)
- Dynamic example generation based on context
- Multi-language property definitions





