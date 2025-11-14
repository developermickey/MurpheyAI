"""
Main training script for the language model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
import wandb
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

from training.models.transformer_model import create_model, MODEL_CONFIGS
from training.data.dataset import TextDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Model trainer."""
    
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Initialize model
        model_size = self.config.get("model_size", "small")
        model_config = MODEL_CONFIGS[model_size]
        self.model = create_model(model_config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 6e-4),
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # Dataset
        dataset = TextDataset(
            data_path=self.config["data_path"],
            seq_len=self.config.get("seq_len", 2048),
        )
        
        # DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            num_workers=4,
        )
        
        # Scheduler
        num_training_steps = len(self.dataloader) * self.config.get("num_epochs", 1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 2000),
            num_training_steps=num_training_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.dataloader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.dataloader, self.scheduler
            )
        
        # Wandb
        if self.config.get("use_wandb", False):
            wandb.init(
                project="murpheyai-training",
                config=self.config,
            )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.config.get("gradient_clip", 1.0) > 0:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("gradient_clip", 1.0),
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.get("log_interval", 100) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({"loss": avg_loss})
                
                if self.config.get("use_wandb", False):
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    })
        
        return total_loss / len(self.dataloader)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save({
            "epoch": epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get("num_epochs", 1)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{num_epochs}")
            
            avg_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if epoch % self.config.get("save_interval", 1) == 0:
                self.save_checkpoint(epoch, avg_loss)
        
        logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()

