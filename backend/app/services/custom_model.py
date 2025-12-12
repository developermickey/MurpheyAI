"""
Utilities to load and generate from custom-trained Transformer checkpoints.

Expected layout under backend/models/<model_name>:
- tokenizer.json            : tokenizer file (tokenizers library)
- checkpoint_epoch_*.pt or checkpoint.pt : torch checkpoint with keys:
    - "model_state_dict"
    - "config" (training config with model_size, etc.)

This mirrors the simple TransformerLM used in training.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from tokenizers import Tokenizer


class TransformerBlock(nn.Module):
    """Single decoder block with causal self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.ln_2(x)
        x = x + self.mlp(h)
        return x


class TransformerLM(nn.Module):
    """Causal LM used for custom checkpoints."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float("-inf"), diagonal=1),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        attn_mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


MODEL_CONFIGS = {
    "small": {
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.1,
    },
    "medium": {
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 8,
        "dropout": 0.1,
    },
    "large": {
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1,
    },
}


def load_custom_model(model_dir: Path, device: torch.device) -> Tuple[TransformerLM, Tokenizer]:
    """Load custom checkpoint and tokenizer from a directory."""
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # tokenizer
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()

    # checkpoint
    ckpts = sorted(model_dir.glob("checkpoint_epoch_*.pt"), reverse=True)
    if not ckpts:
        ckpt = model_dir / "checkpoint.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    else:
        ckpt = ckpts[0]

    state = torch.load(ckpt, map_location=device)
    cfg = state.get("config", {})
    model_size = cfg.get("model_size", "small")
    model_cfg = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["small"]).copy()
    model_cfg["vocab_size"] = vocab_size

    model = TransformerLM(**model_cfg)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer


def _sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    """Sample a token id from logits (1, vocab)."""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1))
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id)


def generate_custom(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    device: Optional[torch.device] = None,
) -> str:
    """Greedy/temperature sampling generation for the custom model."""
    device = device or torch.device("cpu")
    model.eval()

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids: List[int] = encoded.ids[: model.max_seq_len]

    for _ in range(max_tokens):
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_tensor)
            next_logits = logits[0, -1, :]
            next_id = _sample_next_token(next_logits, temperature)
        input_ids.append(next_id)

        if len(input_ids) >= model.max_seq_len:
            break

    return tokenizer.decode(input_ids)

