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

After training:
1. Merge LoRA weights: `model.merge_and_unload()`
2. Convert to GGUF for llama.cpp
3. Test with `tools/demo/constrained_generate.py`
