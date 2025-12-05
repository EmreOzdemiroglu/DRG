import os
import pytest
from drg.schema import Entity, Relation, DRGSchema
from drg.extract import extract_triples
from drg.graph import KG

def test_end_to_end():
    # Check API key (skip if no API key)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this test.")
    
    # Use a cheap/fast model for testing - environment variable set et (DSPy otomatik okur)
    model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    os.environ["DRG_MODEL"] = model
    
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")]
    )
    text = "Apple released the iPhone 16 in September 2025."
    triples = extract_triples(text, schema)
    # Remove duplicates
    triples = list(dict.fromkeys(triples))
    kg = KG.from_triples(triples)
    js = kg.to_json()
    assert "Apple" in js and "iPhone 16" in js
