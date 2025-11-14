"""
Dataset class for training.
"""
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str = "./tokenizer/tokenizer.json",
        seq_len: int = 2048,
    ):
        self.seq_len = seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load data
        self.texts = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            files = [data_path]
        else:
            files = list(data_path.glob("*.jsonl"))
        
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.texts.append(data.get("text", ""))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        input_ids = encoded.ids
        
        # Truncate or pad
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
        else:
            input_ids = input_ids + [0] * (self.seq_len - len(input_ids))
        
        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

