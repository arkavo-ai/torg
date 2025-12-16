# Training Issues & Next Steps

## Completed
- Successfully trained LoRA adapter on HuggingFace AutoTrain
- Model: `mistralai/Ministral-8B-Instruct-2410` (base)
- Output: `Arkavo/torg-ministral-8b-lora` (LoRA adapter)
- Training loss: 4.54 → 1.93 → 2.77 (final average)
- Dataset: 1,072 examples in JSONL format

## Issues Encountered

### 1. Parquet Version Mismatch
- **Problem**: Local pyarrow 22.0.0 creates parquet files incompatible with AutoTrain's older pyarrow
- **Error**: `OSError: Repetition level histogram size mismatch`
- **Solution**: Use JSONL format instead of parquet (see `fix_dataset.py`)

### 2. Ministral-3 2512 Not Supported
- **Problem**: Tried to use newer `Ministral-3-8B-Instruct-2512` model
- **Error**: `Tokenizer class TokenizersBackend does not exist` and `model type mistral3 not recognized`
- **Cause**: AutoTrain's Transformers version is too old for Ministral-3 models
- **Solution**: Fell back to `Ministral-8B-Instruct-2410`

### 3. Local Testing - Memory Pressure
- **Problem**: 8B model too large for local Mac testing
- MPS (Metal) loading hangs due to memory pressure
- CPU loading requires ~32GB RAM
- 4-bit quantization (bitsandbytes) requires CUDA, not available on Mac

### 4. HF Inference Endpoint - LoRA Not Loaded
- **Problem**: Created Inference Endpoint but LoRA adapter wasn't applied
- **Cause**: Text Generation Inference (TGI) doesn't load PEFT/LoRA adapters directly
- **Output**: Model just echoed prompts instead of generating TØR-G tokens
- **Solution needed**: Merge LoRA into base model first

## Next Steps

### Option A: Merge Locally (needs ~32GB RAM)
```bash
cd tools/train
python merge_and_export.py --checkpoint Arkavo/torg-ministral-8b-lora
```
This will:
1. Load base model + LoRA adapter
2. Merge weights
3. Convert to GGUF for llama.cpp

### Option B: Merge on HuggingFace Space
Create a Space with GPU to merge the model in the cloud, then download the merged GGUF.

### Option C: Retrain with Merged Output
Modify AutoTrain to output a merged model instead of LoRA adapter:
- Remove `--peft` flag
- Remove `--quantization` flag
- This trains full model (slower, more expensive)

## Files Modified
- `fix_dataset.py` - Now uses JSONL format instead of parquet
- `run_autotrain.py` - Uses `Ministral-8B-Instruct-2410` (2512 not supported)

## Resources
- Trained LoRA: https://huggingface.co/Arkavo/torg-ministral-8b-lora
- Dataset: https://huggingface.co/datasets/Arkavo/torg-dataset
- Base model: https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
