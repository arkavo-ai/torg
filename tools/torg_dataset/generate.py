#!/usr/bin/env python3
"""Main CLI for generating TØR-G training datasets.

Usage:
    python generate.py --output dataset.jsonl
    python generate.py --output dataset.jsonl --upload
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

from templates import generate_template_examples, TEMPLATES
from golden import get_golden_examples


def validate_with_torg_core(completion: str) -> bool:
    """Validate a token sequence with torg-core (if available)."""
    # TODO: Add PyO3 bindings or subprocess validation
    # For now, do basic structural validation
    tokens = [int(t) for t in completion.split()]

    # Basic validation: must have at least OUTPUT_DECL (42) + Id
    if 42 not in tokens:
        return False

    # Must end with an Id token (45+)
    if tokens[-1] < 45:
        return False

    return True


def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicate examples based on prompt."""
    seen = set()
    unique = []
    for example in examples:
        if example["prompt"] not in seen:
            seen.add(example["prompt"])
            unique.append(example)
    return unique


def generate_dataset(
    max_template_per: int = 80,
    include_golden: bool = True,
    include_llm: bool = False,
    llm_count: int = 100,
    validate: bool = True,
) -> List[Dict]:
    """Generate the complete training dataset."""
    examples = []

    # Stage 1: Template expansion
    print("Stage 1: Generating template-expanded examples...")
    template_examples = generate_template_examples(max_per_template=max_template_per)
    examples.extend(template_examples)
    print(f"  Generated {len(template_examples)} template examples")

    # Stage 2: Golden examples
    if include_golden:
        print("Stage 2: Adding golden examples...")
        golden_examples = get_golden_examples()
        examples.extend(golden_examples)
        print(f"  Added {len(golden_examples)} golden examples")

    # Stage 3: LLM-augmented examples (optional)
    if include_llm:
        print("Stage 3: Generating LLM-augmented examples...")
        try:
            from llm_augment import generate_augmented_examples
            llm_examples = generate_augmented_examples(count=llm_count)
            examples.extend(llm_examples)
            print(f"  Added {len(llm_examples)} LLM-augmented examples")
        except Exception as e:
            print(f"  Skipped LLM augmentation: {e}")

    # Deduplicate
    examples = deduplicate(examples)
    print(f"After deduplication: {len(examples)} examples")

    # Validate
    if validate:
        print("Validating examples...")
        valid = []
        invalid_count = 0
        for example in examples:
            if validate_with_torg_core(example["completion"]):
                valid.append(example)
            else:
                invalid_count += 1
        if invalid_count > 0:
            print(f"  Removed {invalid_count} invalid examples")
        examples = valid

    return examples


def write_jsonl(examples: List[Dict], output_path: Path) -> None:
    """Write examples to a JSONL file."""
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(examples)} examples to {output_path}")


def upload_to_hub(jsonl_path: Path, repo_id: str = "Arkavo/torg-dataset") -> None:
    """Upload dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Uploading to {repo_id}...")

    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]

    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id)
    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


def print_stats(examples: List[Dict]) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    total = len(examples)
    print(f"Total examples: {total}")

    # Token sequence lengths
    lengths = [len(ex["completion"].split()) for ex in examples]
    print(f"Token sequence length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    # Prompt lengths
    prompt_lens = [len(ex["prompt"]) for ex in examples]
    print(f"Prompt length (chars): min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.1f}")

    # Sample examples
    print("\nSample examples:")
    for example in examples[:3]:
        print(f"  Prompt: {example['prompt']}")
        print(f"  Tokens: {example['completion'][:60]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate TØR-G training dataset")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("torg_dataset.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=80,
        help="Maximum examples per template"
    )
    parser.add_argument(
        "--no-golden",
        action="store_true",
        help="Exclude golden examples"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Include LLM-augmented examples (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--llm-count",
        type=int,
        default=100,
        help="Number of LLM-augmented examples to generate"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub (Arkavo/torg-dataset)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="Arkavo/torg-dataset",
        help="HuggingFace repo ID for upload"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics"
    )

    args = parser.parse_args()

    # Generate dataset
    examples = generate_dataset(
        max_template_per=args.max_per_template,
        include_golden=not args.no_golden,
        include_llm=args.llm,
        llm_count=args.llm_count,
        validate=not args.no_validate,
    )

    # Write to file
    write_jsonl(examples, args.output)

    # Print stats
    if args.stats:
        print_stats(examples)

    # Upload if requested
    if args.upload:
        upload_to_hub(args.output, args.repo)

    print("\nDone!")


if __name__ == "__main__":
    main()
