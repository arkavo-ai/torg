#!/usr/bin/env python3
"""Recreate dataset fresh using JSONL format to avoid parquet version issues.

The HuggingFace AutoTrain environment uses an older pyarrow version that can't
read parquet files created by newer pyarrow (22+). Using JSONL avoids this.
"""

import os
import json
from huggingface_hub import HfApi


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set")
        return

    api = HfApi(token=token)

    # Load the original generated data
    data_file = "../torg_dataset/torg_dataset.jsonl"

    print(f"Loading from {data_file}...")
    examples = []
    with open(data_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"  Loaded {len(examples)} examples")

    # Create text column
    template = """### Policy:
{prompt}

### TØR-G Token Sequence:
{completion}"""

    for ex in examples:
        ex["text"] = template.format(prompt=ex["prompt"], completion=ex["completion"])

    print(f"\nExample:")
    print("-" * 60)
    print(examples[0]["text"])
    print("-" * 60)

    # Write JSONL file
    jsonl_file = "/tmp/train.jsonl"
    with open(jsonl_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {jsonl_file}")

    # Create the repo (delete if exists to ensure clean state)
    try:
        api.delete_repo("Arkavo/torg-dataset", repo_type="dataset")
        print("Deleted existing dataset repo")
    except Exception:
        pass  # Repo doesn't exist

    api.create_repo("Arkavo/torg-dataset", repo_type="dataset", exist_ok=True)

    # Upload JSONL file (avoids parquet version issues)
    api.upload_file(
        path_or_fileobj=jsonl_file,
        path_in_repo="train.jsonl",
        repo_id="Arkavo/torg-dataset",
        repo_type="dataset",
        token=token,
    )
    print("Uploaded JSONL file")

    # Create dataset card
    readme = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# TØR-G Training Dataset

Dataset for fine-tuning LLMs on TØR-G token generation.

- 1,072 examples
- Columns: prompt, completion, text
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id="Arkavo/torg-dataset",
        repo_type="dataset",
        token=token,
    )
    print("Done! Dataset at: https://huggingface.co/datasets/Arkavo/torg-dataset")


if __name__ == "__main__":
    main()
