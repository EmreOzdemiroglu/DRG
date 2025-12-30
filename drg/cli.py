#!/usr/bin/env python3
"""CLI interface for DRG - Declarative Relationship Generation"""
import argparse
import sys
from pathlib import Path

from .schema import Entity, Relation, DRGSchema, load_schema_from_json
from .extract import extract_typed
from .graph import KG


def create_default_schema():
    """Varsayılan şema: Company -> Product"""
    return DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")]
    )


def main():
    parser = argparse.ArgumentParser(
        description="DRG - Declarative Relationship Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  drg extract input.txt -o output.json
  drg extract input.txt -o output.json --schema custom_schema.json
  echo "Apple released iPhone 16" | drg extract - -o output.json
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input text file or '-' for stdin"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="-",
        help="Output JSON file (default: stdout, or specify path like 'outputs/output.json')"
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        help="Custom schema JSON file (optional, uses default Company->Product if not provided)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model identifier. Examples: 'openai/gpt-4o-mini' (cloud, needs API key), 'ollama_chat/llama3' (local, no API key). Default: from DRG_MODEL env or 'openai/gpt-4o-mini'"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM (required for cloud models, not needed for local models like Ollama)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL (optional)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        text = input_path.read_text(encoding="utf-8")
    
    # Load schema
    if args.schema:
        try:
            schema = load_schema_from_json(args.schema)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid schema file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load schema: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        schema = create_default_schema()
    
    # Set environment variables for automatic LLM configuration (DSPy otomatik okur)
    import os
    if args.model:
        os.environ["DRG_MODEL"] = args.model
    if args.api_key:
        # Set appropriate API key env var based on model
        model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        if "gemini" in model.lower():
            # Different SDK/adapters use different env var names for Gemini.
            # Keep both to be robust (LiteLLM commonly reads GOOGLE_API_KEY).
            os.environ["GEMINI_API_KEY"] = args.api_key
            os.environ["GOOGLE_API_KEY"] = args.api_key
        elif "anthropic" in model.lower() or "claude" in model.lower():
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["DRG_BASE_URL"] = args.base_url
    if args.temperature != 0.0:
        os.environ["DRG_TEMPERATURE"] = str(args.temperature)
    
    # Warn if using cloud model without API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    if not api_key and not model.startswith("ollama"):
        print("Warning: No API key found. Cloud models require an API key.", file=sys.stderr)
        print("For local models, use: --model ollama_chat/llama3 --base-url http://localhost:11434", file=sys.stderr)
    
<<<<<<< HEAD
    # Determine output format
    inferred_format = None
    if args.output != "-" and args.output.lower().endswith("_kg.json"):
        inferred_format = "enhancedkg"
    output_format = args.output_format or inferred_format or "legacy"
    
=======
>>>>>>> a4118681b584e40e8595d2d058b94dc61682c5ce
    # Extract
    try:
        entities_typed, triples = extract_typed(text, schema)
        # Remove duplicates
        triples = list(dict.fromkeys(triples))
        kg = KG.from_typed(entities_typed, triples)
        output_json = kg.to_json()
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Write output
    if args.output == "-":
        print(output_json)
    else:
        output_path = Path(args.output)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Knowledge graph written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

