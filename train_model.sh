#!/bin/bash
# Quick script to train your custom model

echo "🚀 Starting Custom Model Training for MurpheyAI"
echo ""

cd "$(dirname "$0")/training" || exit 1

# Check if dependencies are installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "📦 Installing training dependencies..."
    pip install -r requirements.txt
fi

# Run training
echo "🎯 Starting training..."
python3 train_my_model.py

echo ""
echo "✅ Training complete! Check the output above for next steps."

