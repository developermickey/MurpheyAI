#!/usr/bin/env python3
"""
Simple script to train your own custom model for MurpheyAI.

This script will:
1. Prepare sample training data
2. Train a tokenizer
3. Train a small language model
4. Export it to the backend

Usage:
    python train_my_model.py
"""

import json
import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tokenizer.train_tokenizer import train_tokenizer
    from data.data_processor import DataProcessor
    from scripts.train import Trainer
    from scripts.export_model import export_checkpoint
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install training dependencies:")
    print("  cd training && pip install -r requirements.txt")
    sys.exit(1)


def create_sample_data(output_file: Path):
    """Create sample training data if none exists."""
    if output_file.exists():
        print(f"✓ Sample data already exists at {output_file}")
        return
    
    print(f"Creating sample training data at {output_file}...")
    
    # Sample conversational data
    sample_data = [
        {"text": "Hello! How can I help you today? I'm MurpheyAI, your AI assistant."},
        {"text": "What is artificial intelligence? AI is the simulation of human intelligence by machines."},
        {"text": "How does machine learning work? Machine learning uses algorithms to learn patterns from data."},
        {"text": "What is Python? Python is a high-level programming language known for its simplicity."},
        {"text": "Explain neural networks. Neural networks are computing systems inspired by biological neural networks."},
        {"text": "What is deep learning? Deep learning uses neural networks with multiple layers to learn complex patterns."},
        {"text": "How do transformers work? Transformers use attention mechanisms to process sequences of data."},
        {"text": "What is natural language processing? NLP is a field of AI that focuses on understanding human language."},
        {"text": "Explain computer vision. Computer vision enables machines to interpret and understand visual information."},
        {"text": "What is reinforcement learning? Reinforcement learning is training agents to make decisions through rewards."},
        {"text": "How does a chatbot work? Chatbots use NLP and machine learning to understand and respond to user messages."},
        {"text": "What is data science? Data science combines statistics, programming, and domain expertise to extract insights."},
        {"text": "Explain big data. Big data refers to extremely large datasets that require special tools to process."},
        {"text": "What is cloud computing? Cloud computing delivers computing services over the internet."},
        {"text": "How does encryption work? Encryption converts data into a secure format that can only be read with a key."},
        {"text": "What is blockchain? Blockchain is a distributed ledger technology that ensures data integrity."},
        {"text": "Explain quantum computing. Quantum computing uses quantum mechanics to perform computations."},
        {"text": "What is the Internet of Things? IoT connects everyday devices to the internet for data exchange."},
        {"text": "How does cybersecurity work? Cybersecurity protects systems and data from digital attacks."},
        {"text": "What is software engineering? Software engineering is the application of engineering principles to software development."},
    ]
    
    # Add more variations
    for i in range(5):
        for item in sample_data[:10]:  # Duplicate first 10 items with variations
            sample_data.append({
                "text": item["text"] + " This is an important concept in modern technology."
            })
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Created {len(sample_data)} sample training examples")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  Training Your Custom MurpheyAI Model")
    print("=" * 60)
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    raw_data_file = data_dir / "raw" / "sample_data.jsonl"
    processed_data_dir = data_dir / "processed"
    tokenizer_dir = base_dir / "tokenizer"
    checkpoint_dir = base_dir / "checkpoints"
    backend_models_dir = base_dir.parent / "backend" / "models"
    
    # Step 1: Create sample data
    print("Step 1: Preparing training data...")
    create_sample_data(raw_data_file)
    
    # Step 2: Process data
    print("\nStep 2: Processing and cleaning data...")
    processor = DataProcessor(
        input_dir=str(data_dir / "raw"),
        output_dir=str(processed_data_dir)
    )
    processor.process_all()
    
    # Step 3: Train tokenizer
    print("\nStep 3: Training tokenizer...")
    processed_files = list(processed_data_dir.glob("*.jsonl"))
    if not processed_files:
        print("❌ No processed data files found!")
        return
    
    train_tokenizer(
        data_files=[str(f) for f in processed_files],
        vocab_size=10000,  # Smaller vocab for faster training
        output_dir=str(tokenizer_dir)
    )
    print("✓ Tokenizer trained and saved")
    
    # Step 4: Train model
    print("\nStep 4: Training model (this may take a while)...")
    print("Note: This is a small model for demonstration. For production, use more data and epochs.")
    
    # Create training config
    config = {
        "model_size": "small",
        "data_path": str(processed_data_dir),
        "tokenizer_path": str(tokenizer_dir / "tokenizer.json"),
        "seq_len": 512,  # Smaller for faster training
        "batch_size": 2,
        "num_epochs": 1,  # Start with 1 epoch
        "learning_rate": 6e-4,
        "warmup_steps": 100,
        "gradient_clip": 1.0,
        "checkpoint_dir": str(checkpoint_dir),
        "save_interval": 1,
        "log_interval": 10,
        "use_wandb": False,
    }
    
    # Save config
    import yaml
    config_file = base_dir / "configs" / "my_model_config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    # Train
    try:
        trainer = Trainer(str(config_file))
        trainer.train()
        print("✓ Model training completed")
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("Note: Make sure you have PyTorch and other dependencies installed.")
        return
    
    # Step 5: Export to backend
    print("\nStep 5: Exporting model to backend...")
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"), reverse=True)
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return
    
    latest_checkpoint = checkpoint_files[0]
    model_name = "my-custom-model"
    output_model_dir = backend_models_dir / model_name
    
    export_checkpoint(
        checkpoint_path=str(latest_checkpoint),
        tokenizer_path=str(tokenizer_dir / "tokenizer.json"),
        output_dir=str(output_model_dir),
        model_name=model_name
    )
    
    print("\n" + "=" * 60)
    print("  ✓ Training Complete!")
    print("=" * 60)
    print(f"\nYour model is ready at: {output_model_dir}")
    print("\nTo use your custom model:")
    print("1. Update backend/.env:")
    print(f"   MODEL_PATH=./models")
    print(f"   MODEL_NAME={model_name}")
    print("\n2. Restart the backend server")
    print("\n3. Your chat will now use your custom model!")


if __name__ == "__main__":
    main()

