#!/usr/bin/env python3
"""TØR-G LoRA fine-tuning script for Ministral models.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_with_unsloth(config: dict, dry_run: bool = False):
    """Train using Unsloth (recommended for speed)."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError as e:
        print(f"Error: Missing dependencies. Install with:")
        print("  pip install unsloth trl transformers datasets")
        sys.exit(1)

    print("=" * 60)
    print("TØR-G LoRA Fine-Tuning with Unsloth")
    print("=" * 60)

    model_name = config["model"]["name"]
    local_path = config["model"].get("local_path")

    print(f"\nLoading model: {model_name}")
    if local_path and Path(local_path).exists():
        print(f"  Using local path: {local_path}")
        model_name = local_path

    if dry_run:
        print("[DRY RUN] Would load model and start training")
        return

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=config["quantization"]["load_in_4bit"],
        dtype=None,  # Auto-detect
    )

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
        random_state=42,
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset_name = config["dataset"]["name"]
    local_dataset = config["dataset"].get("local_path")

    if local_dataset and Path(local_dataset).exists():
        dataset = load_dataset("json", data_files=local_dataset, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    print(f"  Dataset size: {len(dataset)} examples")

    # Format prompt template
    template = config["dataset"]["template"]
    prompt_field = config["dataset"]["prompt_field"]
    completion_field = config["dataset"]["completion_field"]

    def format_example(example):
        return template.format(
            prompt=example[prompt_field],
            completion=example[completion_field]
        )

    # Training arguments
    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        logging_steps=config["output"]["logging_steps"],
        save_strategy=config["output"]["save_strategy"],
        save_total_limit=config["output"]["save_total_limit"],
        optim=config["training"]["optimizer"],
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_example,
        max_seq_length=config["training"]["max_seq_length"],
        args=training_args,
    )

    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\nSaving model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nTraining complete! Model saved to {output_dir}/final")


def train_with_peft(config: dict, dry_run: bool = False, force_cpu: bool = False):
    """Train using standard PEFT (fallback)."""
    # If forcing CPU, disable MPS before importing torch
    if force_cpu:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoConfig,
            TrainingArguments,
            Trainer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
        import torch

        # Disable MPS if forcing CPU
        if force_cpu and hasattr(torch.backends, "mps"):
            torch.backends.mps.is_available = lambda: False

    except ImportError as e:
        print(f"Error: Missing dependencies. Install with:")
        print("  pip install transformers peft datasets bitsandbytes accelerate")
        sys.exit(1)

    print("=" * 60)
    print("TØR-G LoRA Fine-Tuning with PEFT")
    print("=" * 60)

    model_name = config["model"]["name"]
    local_path = config["model"].get("local_path")

    print(f"\nLoading model: {model_name}")
    if local_path and Path(local_path).exists():
        print(f"  Using local path: {local_path}")
        model_name = local_path

    if dry_run:
        print("[DRY RUN] Would load model and start training")
        return

    # Detect device
    import platform
    is_mac = platform.system() == "Darwin"
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Force CPU if requested
    if force_cpu:
        print("  Forcing CPU training (--cpu flag set)")
        has_mps = False
        has_cuda = False

    if is_mac and not force_cpu:
        print(f"  Running on macOS (MPS available: {has_mps})")
        # No quantization on Mac (bitsandbytes doesn't support MPS)
        bnb_config = None
        # Load to CPU first, we'll handle device placement later
        device_map = {"": "cpu"}
        torch_dtype = torch.float16
    elif has_cuda:
        print(f"  Running on CUDA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["quantization"]["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        )
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        print(f"  Running on CPU")
        bnb_config = None
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    # Check if this is a Ministral 3 model (multimodal but text-only usage)
    is_ministral3 = "Ministral-3" in model_name or "ministral-3" in model_name.lower()

    # Load model
    print("  Loading model (this may take a while)...")
    if is_ministral3:
        # Ministral 3 uses Mistral3ForConditionalGeneration
        # Vision encoder is unused when no images provided (text-only fine-tuning)
        print("  Detected Ministral 3 model - using Mistral3ForConditionalGeneration")
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        if device_map:
            load_kwargs["device_map"] = device_map

        model = Mistral3ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        # Move to MPS if on Mac and MPS is available
        if is_mac and has_mps:
            print("  Moving model to MPS (this may take a moment)...")
            try:
                # Try moving to MPS
                model = model.to("mps")
                print("  Successfully moved to MPS")
            except RuntimeError as e:
                if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                    print(f"  MPS failed: {e}")
                    print("  Falling back to CPU training")
                    has_mps = False
                    # Model stays on CPU
                else:
                    raise

        # Use AutoProcessor which handles the tokenizer
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    else:
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        if device_map:
            load_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if is_mac and has_mps:
            print("  Moving model to MPS (this may take a moment)...")
            try:
                model = model.to("mps")
                print("  Successfully moved to MPS")
            except RuntimeError as e:
                if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                    print(f"  MPS failed: {e}")
                    print("  Falling back to CPU training")
                    has_mps = False
                else:
                    raise

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("\nLoading dataset...")
    dataset_name = config["dataset"]["name"]
    local_dataset = config["dataset"].get("local_path")

    if local_dataset and Path(local_dataset).exists():
        dataset = load_dataset("json", data_files=local_dataset, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")

    print(f"  Dataset size: {len(dataset)} examples")

    # Format and tokenize
    template = config["dataset"]["template"]
    prompt_field = config["dataset"]["prompt_field"]
    completion_field = config["dataset"]["completion_field"]
    max_length = config["training"]["max_seq_length"]

    def tokenize(example):
        text = template.format(
            prompt=example[prompt_field],
            completion=example[completion_field]
        )
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training
    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Determine device for training
    use_cpu = not has_cuda and not has_mps
    if use_cpu:
        print("\n  Training on CPU (this will be slow but should work)")

    # Adjust precision settings for CPU
    use_fp16 = config["training"]["fp16"] and not use_cpu
    use_bf16 = config["training"]["bf16"] and not use_cpu

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config["output"]["logging_steps"],
        save_strategy=config["output"]["save_strategy"],
        save_total_limit=config["output"]["save_total_limit"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        report_to="none",
        use_cpu=use_cpu,
        no_cuda=use_cpu,  # Disable CUDA when using CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print("\nSaving model...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nTraining complete! Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser(description="TØR-G LoRA fine-tuning")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--backend",
        choices=["unsloth", "peft"],
        default="unsloth",
        help="Training backend (unsloth recommended)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without training"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (slower but avoids MPS memory issues)"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print("Configuration:")
        print(yaml.dump(config, default_flow_style=False))

    if args.backend == "unsloth":
        train_with_unsloth(config, dry_run=args.dry_run)
    else:
        train_with_peft(config, dry_run=args.dry_run, force_cpu=args.cpu)


if __name__ == "__main__":
    main()
