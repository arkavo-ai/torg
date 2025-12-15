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

# Run training (will try MPS, falls back to CPU if memory issues)
python train.py --config config.yaml --backend peft

# Force CPU training (slower but guaranteed to work on Mac)
python train.py --config config.yaml --backend peft --cpu

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
- MPS has buffer size limitations that may prevent loading large models
- If MPS fails, training automatically falls back to CPU
- Use `--cpu` flag to skip MPS attempt and train directly on CPU
- No quantization (bitsandbytes not supported on MPS)
- Batch size 1, gradient accumulation 16
- CPU training is slower but works reliably

### CUDA GPU
- Uses 4-bit QLoRA for memory efficiency
- Can use larger batch sizes

### HuggingFace AutoTrain (Cloud - Recommended)
Use HuggingFace's cloud infrastructure for training without local GPU requirements.

**Option 1: AutoTrain CLI**
```bash
# Install AutoTrain
pip install autotrain-advanced

# Prepare dataset for AutoTrain format
python prepare_autotrain_dataset.py --push

# Run training (uses HF cloud compute)
autotrain llm --config autotrain_config.yaml
```

**Option 2: AutoTrain Web UI**
1. Go to https://huggingface.co/autotrain
2. Create new project → LLM Fine-tuning
3. Select base model: `mistralai/Ministral-8B-Instruct-2410`
4. Upload dataset or use `Arkavo/torg-dataset-autotrain`
5. Configure LoRA: r=16, alpha=32, dropout=0.05
6. Select GPU (A10G or A100) and start training

**Pricing**: ~$2-8/hour depending on GPU. Training ~1000 examples for 3 epochs typically takes 1-2 hours.

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
