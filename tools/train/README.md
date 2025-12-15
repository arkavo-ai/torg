# TØR-G LoRA Fine-Tuning

Fine-tune Ministral-3-8B-Base-2512 for TØR-G token generation.

## Requirements

- Mac Studio M1 with 32GB+ unified memory (or CUDA GPU with 24GB+ VRAM)
- Python 3.10+
- ~35GB disk space for model

## Setup

```bash
# Install dependencies
pip install transformers peft datasets accelerate bitsandbytes trl pyyaml

# For latest Ministral 3 support, install transformers from git
pip install git+https://github.com/huggingface/transformers.git

# Set HuggingFace cache to external drive (optional)
export HF_HOME=/Volumes/SSD/huggingface
```

## Download Model (if not already cached)

```bash
huggingface-cli download mistralai/Ministral-3-8B-Base-2512
```

## Training

```bash
cd tools/train

# Run training
python train.py --config config.yaml --backend peft

# Or dry run to check config
python train.py --config config.yaml --dry-run
```

## Configuration

Edit `config.yaml` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `mistralai/Ministral-3-8B-Base-2512` | Model ID |
| `model.local_path` | (set for cached model) | Local snapshot path |
| `training.batch_size` | 1 | Batch size (keep low for MPS) |
| `training.epochs` | 3 | Training epochs |
| `lora.r` | 16 | LoRA rank |

## Platform Notes

### Mac Studio M1 (32GB)
- Uses MPS backend with fp16
- No quantization (bitsandbytes not supported on MPS)
- Batch size 1, gradient accumulation 16

### CUDA GPU
- Uses 4-bit QLoRA for memory efficiency
- Can use larger batch sizes

## Output

Checkpoints saved to `./output/torg-ministral-8b-lora/`

## Post-Training: Merge & Export

After training completes, merge LoRA and export to GGUF:

```bash
# Merge LoRA + convert to GGUF + quantize
python merge_and_export.py \
    --checkpoint ./output/torg-ministral-8b-lora/final \
    --quantize q4_k_m

# Or step by step:
python merge_and_export.py --checkpoint ./output/torg-ministral-8b-lora/final --skip-gguf  # Merge only
python merge_and_export.py --checkpoint ./output/torg-ministral-8b-lora/final --skip-merge  # GGUF only
```

Output files:
- `./output/torg-ministral-8b-merged/` - Merged HF model
- `./output/torg-ministral-8b-f16.gguf` - Full precision GGUF
- `./output/torg-ministral-8b-q4_k_m.gguf` - Quantized GGUF

## Test

```bash
cd ../demo
python constrained_generate.py --model ../train/output/torg-ministral-8b-q4_k_m.gguf \
    --prompt "Allow if admin OR member"
```
