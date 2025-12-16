#!/usr/bin/env python3
"""Run AutoTrain LoRA fine-tuning on HuggingFace Spaces infrastructure.

Usage:
    python run_autotrain.py

This uses AutoTrain SpaceRunner to train on HF cloud GPUs.
Pricing: ~$1-4/hour depending on GPU selected.
"""

import os
import subprocess
import sys


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Set it with: export HF_TOKEN=your_token")
        sys.exit(1)

    print("=" * 60)
    print("TØR-G AutoTrain SpaceRunner - Cloud Training")
    print("=" * 60)

    # AutoTrain SpaceRunner command
    cmd = [
        "autotrain", "llm",
        "--train",
        "--model", "mistralai/Ministral-8B-Instruct-2410",
        "--data-path", "Arkavo/torg-dataset",
        "--text-column", "text",
        "--train-split", "train",
        "--project-name", "torg-ministral-8b-lora",
        "--username", "Arkavo",
        "--token", token,
        "--push-to-hub",
        # LoRA config
        "--peft",
        "--quantization", "int4",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",
        "--target-modules", "q_proj,k_proj,v_proj,o_proj",
        # Training params
        "--epochs", "3",
        "--batch-size", "2",
        "--gradient-accumulation", "8",
        "--lr", "2e-4",
        "--warmup-ratio", "0.03",
        "--model-max-length", "512",
        "--optimizer", "adamw_torch",
        "--scheduler", "cosine",
        "--weight-decay", "0.01",
        "--mixed-precision", "bf16",
        "--logging-steps", "10",
        # Run on HF Spaces with L4 GPU (try this if A10G fails)
        "--backend", "spaces-l4x1",
    ]

    print(f"\nConfiguration:")
    print(f"  Model: mistralai/Ministral-8B-Instruct-2410")
    print(f"  Dataset: Arkavo/torg-dataset")
    print(f"  LoRA: r=16, alpha=32")
    print(f"  Epochs: 3")
    print(f"  Backend: spaces-l4x1 (L4 GPU)")
    print(f"  Output: https://huggingface.co/Arkavo/torg-ministral-8b-lora")

    print("\n" + "=" * 60)
    print("Submitting training job to HuggingFace Spaces...")
    print("=" * 60 + "\n")

    # Run the command
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Training job submitted!")
        print("Monitor progress at: https://huggingface.co/spaces/Arkavo/autotrain-torg-ministral-8b-lora")
        print("=" * 60)
    else:
        print(f"\nError: Training submission failed with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
