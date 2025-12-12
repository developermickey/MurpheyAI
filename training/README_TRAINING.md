# Train Your Own Custom Model

This guide will help you create and train your own AI model for MurpheyAI.

## 🚀 Quick Start

The easiest way to train your model:

```bash
cd training
pip install -r requirements.txt
python train_my_model.py
```

This will:
1. Create sample training data
2. Process and clean it
3. Train a tokenizer
4. Train a small model
5. Export it to the backend

## 📝 Step-by-Step Guide

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create JSONL files (one JSON per line) in `training/data/raw/`:

```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

### 3. Run Training

**Option A: Automated (Recommended)**
```bash
python train_my_model.py
```

**Option B: Manual Steps**
```bash
# Process data
python -m data.data_processor

# Train tokenizer
python -m tokenizer.train_tokenizer

# Train model
python scripts/train.py --config configs/train_config.yaml

# Export model
python scripts/export_model.py \
    --checkpoint checkpoints/checkpoint_epoch_1.pt \
    --tokenizer tokenizer/tokenizer.json \
    --output-dir ../backend/models/my-model \
    --model-name my-model
```

### 4. Activate Your Model

Edit `backend/.env`:
```env
MODEL_PATH=./models
MODEL_NAME=my-custom-model
```

Restart the backend server.

## 🎯 Model Configuration

Edit `training/configs/train_config.yaml`:

```yaml
model_size: "small"  # small, medium, or large
batch_size: 4
num_epochs: 3        # More epochs = better model (but slower)
learning_rate: 6e-4
seq_len: 2048        # Context length
```

## 💡 Tips for Better Models

1. **More Data**: Add more training examples (1000+ recommended)
2. **Better Data**: Use high-quality, relevant text
3. **More Epochs**: Train for 3-10 epochs
4. **Larger Model**: Use "medium" or "large" for better quality
5. **Domain-Specific**: Train on data relevant to your use case

## 🔧 Troubleshooting

**"Module not found" errors:**
```bash
pip install torch transformers accelerate tokenizers tqdm pyyaml
```

**Out of memory:**
- Reduce `batch_size` to 1 or 2
- Reduce `seq_len` to 512
- Use CPU instead of GPU

**Poor model quality:**
- Add more training data
- Train for more epochs
- Use larger model size
- Check data quality

## 📊 Model Sizes

- **small**: Fast, less accurate (good for testing)
- **medium**: Balanced speed/quality
- **large**: Slower, more accurate (best quality)

## 🎓 Next Steps

- Add conversation data for better chat responses
- Fine-tune on specific domains
- Experiment with hyperparameters
- Monitor training with wandb

For detailed information, see `QUICK_TRAIN.md`.

