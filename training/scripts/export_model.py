"""
Export a trained checkpoint to the backend models directory.

This script takes a training checkpoint and exports it to the format
expected by the backend's custom model loader.

Usage:
    python scripts/export_model.py \
        --checkpoint checkpoints/checkpoint_epoch_1.pt \
        --tokenizer tokenizer/tokenizer.json \
        --output-dir ../backend/models/my-model \
        --model-name my-model
"""
import argparse
import shutil
from pathlib import Path
import torch
import yaml
import json


def export_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str,
    model_name: str,
):
    """
    Export a training checkpoint to backend format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        tokenizer_path: Path to tokenizer.json
        output_dir: Directory to export to (backend/models/<model_name>)
        model_name: Name for the model
    """
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy tokenizer
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    print(f"✓ Copied tokenizer to {output_dir / 'tokenizer.json'}")
    
    # Load checkpoint and save in expected format
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Save checkpoint with expected structure
    # The backend expects:
    # - model_state_dict
    # - config (training config with model_size, etc.)
    
    # Extract model_state_dict
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    else:
        # Assume entire checkpoint is the state dict
        model_state_dict = checkpoint
    
    # Extract config
    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        # Ensure we have model_size
        if "model_size" not in config:
            config["model_size"] = "small"  # default
    else:
        config = {"model_size": "small"}
    
    # Save checkpoint in expected format
    export_checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": model_state_dict,
        "config": config,
        "epoch": checkpoint.get("epoch", 1),
        "loss": checkpoint.get("loss", 0.0),
    }, export_checkpoint_path)
    print(f"✓ Exported checkpoint to {export_checkpoint_path}")
    
    # Create a config.json for reference
    config_json = {
        "model_name": model_name,
        "model_size": config.get("model_size", "small"),
        "tokenizer_path": "tokenizer.json",
        "checkpoint_path": "checkpoint.pt",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    print(f"✓ Created config.json")
    
    print(f"\n✓ Model exported successfully to: {output_dir}")
    print(f"\nTo use this model, set in backend/.env:")
    print(f"  MODEL_PATH=./models")
    print(f"  MODEL_NAME={model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained checkpoint to backend format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (backend/models/<model_name>)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name",
    )
    
    args = parser.parse_args()
    
    export_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

