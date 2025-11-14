"""
Train a BPE tokenizer from scratch.
"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tokenizer(
    data_files: list,
    vocab_size: int = 50000,
    output_dir: str = "./tokenizer"
):
    """Train a BPE tokenizer."""
    logger.info(f"Training tokenizer with vocab size {vocab_size}...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Special tokens
    special_tokens = [
        "<|endoftext|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|unk|>",
        "<|pad|>",
    ]
    
    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
    )
    
    # Train
    tokenizer.train(files=data_files, trainer=trainer)
    
    # Post-processor
    tokenizer.post_processor = BertProcessing(
        ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
        ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
    )
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    logger.info(f"Tokenizer saved to {output_path}")
    return tokenizer


if __name__ == "__main__":
    # Example usage
    data_files = [
        "./data/processed/processed_wikipedia_en.jsonl",
        "./data/processed/processed_github_code.jsonl",
    ]
    
    train_tokenizer(
        data_files=data_files,
        vocab_size=50000,
        output_dir="./tokenizer"
    )

