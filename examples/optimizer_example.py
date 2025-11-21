#!/usr/bin/env python3
"""
Example: Using DSPy Optimizer for KG refinement

This demonstrates how to use the optimize_extractor function
to iteratively improve extraction quality with training examples.
"""

import os
import dspy
from drg import (
    Entity, Relation, DRGSchema, 
    KGExtractor,
    optimize_extractor
)

def main():
    # Environment variable'ları kontrol et ve set et (DSPy otomatik okur)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ API key bulunamadı. GEMINI_API_KEY veya OPENAI_API_KEY set edin.")
        return
    
    model = os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    # Environment variable'ları set et (DSPy otomatik okur)
    os.environ["DRG_MODEL"] = model
    
    # Define schema
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")]
    )
    
    # Create base extractor
    extractor = KGExtractor(schema)
    
    # Create training examples
    # Each example should have: text, entities (JSON string), relations (JSON string)
    training_examples = [
        dspy.Example(
            text="Apple released the iPhone 16 in September 2025.",
            entities='[["Apple", "Company"], ["iPhone 16", "Product"]]',
            relations='[["Apple", "produces", "iPhone 16"]]'
        ).with_inputs("text"),
        dspy.Example(
            text="Samsung produces the Galaxy S24 and Galaxy S25 smartphones.",
            entities='[["Samsung", "Company"], ["Galaxy S24", "Product"], ["Galaxy S25", "Product"]]',
            relations='[["Samsung", "produces", "Galaxy S24"], ["Samsung", "produces", "Galaxy S25"]]'
        ).with_inputs("text"),
        dspy.Example(
            text="Microsoft launched Surface Pro 10 and Surface Laptop 6.",
            entities='[["Microsoft", "Company"], ["Surface Pro 10", "Product"], ["Surface Laptop 6", "Product"]]',
            relations='[["Microsoft", "produces", "Surface Pro 10"], ["Microsoft", "produces", "Surface Laptop 6"]]'
        ).with_inputs("text"),
    ]
    
    print("🔄 Optimizing extractor with training examples...")
    print(f"   Training examples: {len(training_examples)}")
    
    # Optimize the extractor
    try:
        optimized_extractor = optimize_extractor(
            extractor=extractor,
            training_examples=training_examples,
            max_bootstrapped_demos=4,
            max_labeled_demos=16
        )
        
        print("✅ Optimizer başarılı!")
        
        # Test with new text
        test_text = "Google released Pixel 9 and Pixel 9 Pro in October 2024."
        print(f"\n📄 Test metni: {test_text}")
        
        result = optimized_extractor(text=test_text)
        print(f"\n📊 Sonuçlar:")
        print(f"   Entities: {result.entities}")
        print(f"   Relations: {result.relations}")
        
        # Save to JSON file
        from drg.graph import KG
        from drg.optimize import refine_triples
        import json
        
        # Convert to KG format
        entities_typed = []
        if result.entities:
            for item in result.entities:
                if isinstance(item, list) and len(item) >= 2:
                    entities_typed.append((str(item[0]), str(item[1])))
        
        triples = []
        if result.relations:
            for item in result.relations:
                if isinstance(item, list) and len(item) >= 3:
                    triples.append((str(item[0]), str(item[1]), str(item[2])))
        
        triples = refine_triples(triples)
        kg = KG.from_typed(entities_typed, triples)
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/optimized_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(kg.to_json())
        
        print(f"\n💾 Output kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"❌ Optimizer hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

