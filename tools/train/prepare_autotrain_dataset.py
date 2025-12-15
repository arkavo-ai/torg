#!/usr/bin/env python3
"""Add 'text' column to existing dataset for AutoTrain compatibility.

Usage:
    python prepare_autotrain_dataset.py
    python prepare_autotrain_dataset.py --push
"""

import argparse
import os
from datasets import load_dataset


def prepare_dataset(push_to_hub: bool = False):
    """Add 'text' column to Arkavo/torg-dataset for AutoTrain."""
    print("Loading dataset from Arkavo/torg-dataset...")
    dataset = load_dataset("Arkavo/torg-dataset", split="train")

    print(f"  Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")

    # Template for combining prompt and completion
    template = """### Policy:
{prompt}

### TØR-G Token Sequence:
{completion}"""

    def add_text_column(example):
        return {
            "text": template.format(
                prompt=example["prompt"],
                completion=example["completion"]
            )
        }

    print("Adding 'text' column for AutoTrain...")
    # Keep original columns, just add 'text'
    formatted_dataset = dataset.map(add_text_column)

    print(f"  Columns: {formatted_dataset.column_names}")
    print(f"\nExample:")
    print("-" * 60)
    print(formatted_dataset[0]["text"])
    print("-" * 60)

    if push_to_hub:
        print("\nPushing to Arkavo/torg-dataset...")
        token = os.environ.get("HF_TOKEN")
        if token:
            print("  Using HF_TOKEN from environment")
        formatted_dataset.push_to_hub("Arkavo/torg-dataset", token=token)
        print("Done! Dataset updated at: https://huggingface.co/datasets/Arkavo/torg-dataset")
    else:
        print("\nTo push to HuggingFace Hub, run with --push flag")

    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for AutoTrain")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push updated dataset to HuggingFace Hub"
    )
    args = parser.parse_args()

    prepare_dataset(push_to_hub=args.push)


if __name__ == "__main__":
    main()
