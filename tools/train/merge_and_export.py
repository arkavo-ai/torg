#!/usr/bin/env python3
"""Merge LoRA weights and export to GGUF format.

Usage:
    python merge_and_export.py --checkpoint ./output/torg-ministral-8b-lora/final
    python merge_and_export.py --checkpoint ./output/torg-ministral-8b-lora/final --quantize q4_k_m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def merge_lora(checkpoint_path: str, output_path: str, base_model: str = None):
    """Merge LoRA weights into base model."""
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import Mistral3ForConditionalGeneration, AutoProcessor

    print("=" * 60)
    print("Step 1: Merge LoRA weights")
    print("=" * 60)

    # Load LoRA config to get base model
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    base_model = base_model or peft_config.base_model_name_or_path

    print(f"  Base model: {base_model}")
    print(f"  LoRA checkpoint: {checkpoint_path}")

    # Detect if Ministral 3
    is_ministral3 = "Ministral-3" in base_model or "ministral-3" in base_model.lower()

    print("  Loading base model...")
    if is_ministral3:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    print("  Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    print("  Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"  Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # Also save tokenizer/processor
    print("  Saving tokenizer...")
    if is_ministral3:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor.save_pretrained(output_path)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

    print(f"  Merged model saved to: {output_path}")
    return output_path


def convert_to_gguf(merged_path: str, output_gguf: str, llama_cpp_path: str = None):
    """Convert merged model to GGUF format."""
    print("\n" + "=" * 60)
    print("Step 2: Convert to GGUF")
    print("=" * 60)

    # Find or clone llama.cpp
    if llama_cpp_path and Path(llama_cpp_path).exists():
        llama_cpp = Path(llama_cpp_path)
    else:
        llama_cpp = Path("./llama.cpp")
        if not llama_cpp.exists():
            print("  Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/ggerganov/llama.cpp",
                str(llama_cpp)
            ], check=True)

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"  ERROR: {convert_script} not found")
        print("  Please clone llama.cpp manually or specify --llama-cpp-path")
        return None

    print(f"  Converting {merged_path} to GGUF...")
    subprocess.run([
        sys.executable, str(convert_script),
        merged_path,
        "--outfile", output_gguf,
        "--outtype", "f16"
    ], check=True)

    print(f"  GGUF saved to: {output_gguf}")
    return output_gguf


def quantize_gguf(input_gguf: str, output_gguf: str, quant_type: str, llama_cpp_path: str = None):
    """Quantize GGUF model."""
    print("\n" + "=" * 60)
    print(f"Step 3: Quantize to {quant_type}")
    print("=" * 60)

    llama_cpp = Path(llama_cpp_path) if llama_cpp_path else Path("./llama.cpp")

    # Find quantize binary
    quantize_bin = None
    for name in ["llama-quantize", "quantize", "build/bin/llama-quantize"]:
        candidate = llama_cpp / name
        if candidate.exists():
            quantize_bin = candidate
            break

    if not quantize_bin:
        print("  Quantize binary not found. Building llama.cpp...")
        subprocess.run(["make", "-C", str(llama_cpp), "llama-quantize"], check=True)
        quantize_bin = llama_cpp / "llama-quantize"

    if not quantize_bin.exists():
        print(f"  ERROR: Could not find or build llama-quantize")
        print(f"  You can quantize manually: llama-quantize {input_gguf} {output_gguf} {quant_type}")
        return None

    print(f"  Quantizing {input_gguf}...")
    subprocess.run([str(quantize_bin), input_gguf, output_gguf, quant_type], check=True)

    print(f"  Quantized model saved to: {output_gguf}")
    return output_gguf


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and export to GGUF")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to LoRA checkpoint (e.g., ./output/torg-ministral-8b-lora/final)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model (auto-detected from checkpoint if not specified)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--quantize", "-q",
        type=str,
        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
        default=None,
        help="Quantization type (optional)"
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        default=None,
        help="Path to llama.cpp (will clone if not found)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step (use existing merged model)"
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip GGUF conversion"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA
    merged_path = output_dir / "torg-ministral-8b-merged"
    if not args.skip_merge:
        merge_lora(args.checkpoint, str(merged_path), args.base_model)
    else:
        print(f"Skipping merge, using: {merged_path}")

    if args.skip_gguf:
        print("\nSkipping GGUF conversion.")
        print(f"\nMerged model ready at: {merged_path}")
        return

    # Step 2: Convert to GGUF
    gguf_f16 = output_dir / "torg-ministral-8b-f16.gguf"
    convert_to_gguf(str(merged_path), str(gguf_f16), args.llama_cpp_path)

    # Step 3: Quantize (optional)
    final_gguf = gguf_f16
    if args.quantize and args.quantize != "f16":
        quant_gguf = output_dir / f"torg-ministral-8b-{args.quantize}.gguf"
        result = quantize_gguf(str(gguf_f16), str(quant_gguf), args.quantize, args.llama_cpp_path)
        if result:
            final_gguf = quant_gguf

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nFinal GGUF model: {final_gguf}")
    print(f"\nTest with:")
    print(f"  python tools/demo/constrained_generate.py --model {final_gguf}")


if __name__ == "__main__":
    main()
