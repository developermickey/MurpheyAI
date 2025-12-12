# 🎯 Custom Model Training Guide

## Quick Start

To train your own model and use it in MurpheyAI:

### Method 1: One-Command Training (Easiest)

```bash
./train_model.sh
```

### Method 2: Manual Training

```bash
cd training
pip install -r requirements.txt
python train_my_model.py
```

## What Happens

1. **Sample Data Creation**: Creates training examples automatically
2. **Data Processing**: Cleans and prepares the data
3. **Tokenizer Training**: Creates a tokenizer for your model
4. **Model Training**: Trains a small transformer model
5. **Export**: Saves model to `backend/models/my-custom-model/`

## Activating Your Model

After training completes:

1. **Update backend/.env**:
   ```env
   MODEL_PATH=./models
   MODEL_NAME=my-custom-model
   ```

2. **Restart backend**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

3. **Test it**: Open http://localhost:3000 and start chatting!

## Using Your Own Data

1. Create JSONL files in `training/data/raw/`:
   ```json
   {"text": "Your training text here"}
   {"text": "Another example"}
   ```

2. Run the training script - it will automatically use your data

## Model Configuration

Edit `training/configs/train_config.yaml`:

- **model_size**: `small` (fast) | `medium` (balanced) | `large` (best quality)
- **num_epochs**: More epochs = better model (start with 1-3)
- **batch_size**: Reduce if you get memory errors
- **seq_len**: Context length (512-2048)

## Tips

✅ **Better Results:**
- Add 1000+ training examples
- Use high-quality, relevant text
- Train for 3-10 epochs
- Use "medium" or "large" model size

⚠️ **Troubleshooting:**
- Out of memory? Reduce `batch_size` and `seq_len`
- Poor quality? Add more data and train longer
- Import errors? Run `pip install -r requirements.txt`

## Files Created

After training, you'll have:
- `backend/models/my-custom-model/checkpoint.pt` - Your trained model
- `backend/models/my-custom-model/tokenizer.json` - Your tokenizer
- `backend/models/my-custom-model/config.json` - Model config

## Next Steps

- Add domain-specific training data
- Fine-tune on conversation data
- Experiment with different hyperparameters
- Monitor training progress

For detailed information, see `training/README_TRAINING.md`

