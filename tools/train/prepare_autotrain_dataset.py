#!/usr/bin/env python3
"""Prepare dataset for AutoTrain by converting prompt/completion to text format.

Usage:
    python prepare_autotrain_dataset.py
    python prepare_autotrain_dataset.py --push
"""

import argparse
from datasets import load_dataset


def prepare_dataset(push_to_hub: bool = False):
    """Convert Arkavo/torg-dataset to AutoTrain format."""
    print("Loading dataset from Arkavo/torg-dataset...")
    dataset = load_dataset("Arkavo/torg-dataset", split="train")

    print(f"  Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")

    # Template for combining prompt and completion
    template = """### Policy:
{prompt}

### TØR-G Token Sequence:
{completion}"""

    def format_for_autotrain(example):
        return {
            "text": template.format(
                prompt=example["prompt"],
                completion=example["completion"]
            )
        }

    print("Converting to AutoTrain format...")
    formatted_dataset = dataset.map(format_for_autotrain, remove_columns=["prompt", "completion"])

    print(f"  New columns: {formatted_dataset.column_names}")
    print(f"\nExample:")
    print("-" * 60)
    print(formatted_dataset[0]["text"])
    print("-" * 60)

    if push_to_hub:
        print("\nPushing to Arkavo/torg-dataset-autotrain...")
        formatted_dataset.push_to_hub("Arkavo/torg-dataset-autotrain")
        print("Done! Dataset available at: https://huggingface.co/datasets/Arkavo/torg-dataset-autotrain")
    else:
        print("\nTo push to HuggingFace Hub, run with --push flag")

    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for AutoTrain")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push formatted dataset to HuggingFace Hub"
    )
    args = parser.parse_args()

    prepare_dataset(push_to_hub=args.push)


if __name__ == "__main__":
    main()
