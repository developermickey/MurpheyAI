# Quick Guide: Train Your Own Model

This guide will help you train your own custom AI model for MurpheyAI.

## Prerequisites

1. **Python 3.11+** installed
2. **PyTorch** installed (for training)
3. **Training dependencies** installed

## Installation

```bash
cd training
pip install -r requirements.txt
```

## Quick Start (Easiest Method)

Run the automated training script:

```bash
cd training
python train_my_model.py
```

This script will:
- ✅ Create sample training data
- ✅ Process and clean the data
- ✅ Train a tokenizer
- ✅ Train a small language model
- ✅ Export it to the backend

## Using Your Own Data

1. **Prepare your data:**
   - Create JSONL files (one JSON object per line)
   - Each line should have: `{"text": "your training text here"}`
   - Place files in `training/data/raw/`

2. **Process your data:**
   ```bash
   cd training
   python -m data.data_processor
   ```

3. **Train tokenizer:**
   ```bash
   python -m tokenizer.train_tokenizer
   ```

4. **Train model:**
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

5. **Export model:**
   ```bash
   python scripts/export_model.py \
       --checkpoint checkpoints/checkpoint_epoch_1.pt \
       --tokenizer tokenizer/tokenizer.json \
       --output-dir ../backend/models/my-model \
       --model-name my-model
   ```

## Activate Your Model

1. **Update backend/.env:**
   ```env
   MODEL_PATH=./models
   MODEL_NAME=my-model
   ```

2. **Restart backend:**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

3. **Test in frontend:**
   - Open http://localhost:3000
   - Start a chat - it will use your custom model!

## Model Sizes

You can choose different model sizes in `configs/train_config.yaml`:

- **small**: 256 embed dim, 4 layers (fastest, less accurate)
- **medium**: 512 embed dim, 8 layers (balanced)
- **large**: 768 embed dim, 12 layers (slowest, most accurate)

## Tips for Better Models

1. **More data = Better model**: Add more training examples
2. **More epochs**: Increase `num_epochs` in config (but training takes longer)
3. **Larger model**: Use "medium" or "large" model size
4. **Better data**: Use high-quality, domain-specific text
5. **Longer sequences**: Increase `seq_len` for better context understanding

## Troubleshooting

**Error: "No module named 'torch'"**
```bash
pip install torch transformers accelerate
```

**Error: "CUDA out of memory"**
- Reduce `batch_size` in config
- Use smaller `seq_len`
- Use CPU instead of GPU

**Model responses are poor**
- Train for more epochs
- Add more training data
- Use a larger model size

## Next Steps

- Add your own domain-specific training data
- Fine-tune on conversation data
- Experiment with different hyperparameters
- Monitor training with wandb (set `use_wandb: true` in config)

