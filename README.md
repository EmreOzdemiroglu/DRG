# DRG – Declarative Relationship Generation

A DSPy-inspired library for declarative knowledge graph construction using LLMs.

## Features

- **Declarative Schema Definition**: Define entities and relationships using a simple, declarative syntax
- **LLM-Powered Extraction**: Uses DSPy with your choice of LLM (OpenAI, Anthropic, Gemini, etc.)
- **Schema Validation**: Automatically validates extracted entities and relations against your schema
- **JSON Output**: Export knowledge graphs as structured JSON

## Installation

```bash
uv pip install -e .
```

### Dependencies

- `dspy>=2.4.0` - Declarative AI framework
- `litellm>=1.0.0` - LLM abstraction layer

## Quickstart

### Basic Usage

DRG uses **automatic LLM configuration** via environment variables. No manual configuration needed!

#### With Cloud Models (requires API key)

```python
import os
from drg import Entity, Relation, DRGSchema, extract_typed
from drg.graph import KG

# Set environment variables (DSPy automatically reads them)
os.environ["DRG_MODEL"] = "openai/gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = "your-api-key"  # or set in shell

# Define your schema declaratively
schema = DRGSchema(
    entities=[
        Entity("Company"),
        Entity("Product")
    ],
    relations=[
        Relation("produces", "Company", "Product")
    ]
)

# Extract from text - DSPy handles everything automatically!
text = "Apple released the iPhone 16 in September 2025."
entities, triples = extract_typed(text, schema)

# Build knowledge graph
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

#### With Local Models (no API key needed)

```python
import os
from drg import Entity, Relation, DRGSchema, extract_typed
from drg.graph import KG

# Set environment variables for Ollama (local, no API key required)
os.environ["DRG_MODEL"] = "ollama_chat/llama3"
os.environ["DRG_BASE_URL"] = "http://localhost:11434"  # Ollama default URL

# Define your schema declaratively
schema = DRGSchema(
    entities=[
        Entity("Company"),
        Entity("Product")
    ],
    relations=[
        Relation("produces", "Company", "Product")
    ]
)

# Extract from text - DSPy handles everything automatically!
text = "Apple released the iPhone 16 in September 2025."
entities, triples = extract_typed(text, schema)

# Build knowledge graph
kg = KG.from_typed(entities, triples)
print(kg.to_json())
```

**Or set environment variables in your shell:**

```bash
export DRG_MODEL="openai/gpt-4o-mini"
export OPENAI_API_KEY="your-key"
# Then just use extract_typed() - no configuration needed!
```

### CLI Usage

```bash
# Basic usage (cloud model, needs API key in environment)
drg extract input.txt -o output.json

# With local model (Ollama, no API key needed)
drg extract input.txt -o output.json --model "ollama_chat/llama3" --base-url "http://localhost:11434"

# With custom cloud model
drg extract input.txt -o output.json --model "gemini/gemini-2.0-flash-exp" --api-key "your-key"

# From stdin
echo "Apple released iPhone 16" | drg extract - -o output.json
```

### Supported Models

DRG supports any model via LiteLLM. Common formats:

**Cloud Models (require API key):**
- `openai/gpt-4o-mini`
- `openai/gpt-4o`
- `anthropic/claude-3-5-sonnet-20241022`
- `gemini/gemini-2.0-flash-exp`

**Local Models (no API key needed):**
- `ollama_chat/llama3`
- `ollama_chat/deepseek-r1:14b`
- `ollama_chat/mistral`

For local models, make sure Ollama is running: `ollama serve`

See [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for full list.

## Architecture

DRG uses **DSPy signatures** to declaratively define extraction tasks:

1. **Entity Extraction**: LLM extracts entities matching schema types
2. **Relation Extraction**: LLM extracts relationships matching schema relations
3. **Schema Validation**: Results are validated against the declarative schema
4. **DSPy Optimizer**: Iterative refinement using BootstrapFewShot optimizer

This declarative approach allows you to:
- Define schemas without writing extraction logic
- Switch LLM models easily
- Use DSPy optimizers for iterative refinement and improved precision

### DSPy Optimizer

DRG includes a DSPy-style optimizer for relationship refinement:

```python
from drg import KGExtractor, optimize_extractor, Entity, Relation, DRGSchema
import dspy

# Create extractor
schema = DRGSchema(
    entities=[Entity("Company"), Entity("Product")],
    relations=[Relation("produces", "Company", "Product")]
)
extractor = KGExtractor(schema)

# Create training examples
training_examples = [
    dspy.Example(
        text="Apple released iPhone 16",
        entities='[["Apple", "Company"], ["iPhone 16", "Product"]]',
        relations='[["Apple", "produces", "iPhone 16"]]'
    ).with_inputs("text"),
    # ... more examples
]

# Optimize
optimized_extractor = optimize_extractor(
    extractor=extractor,
    training_examples=training_examples
)
```

See `examples/optimizer_example.py` for a complete example.

## Testing

```bash
# Run tests (requires API key)
export OPENAI_API_KEY="your-key"
pytest -q
```

## License

MIT
