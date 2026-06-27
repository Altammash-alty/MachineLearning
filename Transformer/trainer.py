import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """
    A production-grade Trainer for neural networks in PyTorch.
    Handles device selection, logging, metrics, checkpoint saving, and early stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        gradient_clip_val: Optional[float] = 1.0,
        early_stopping_patience: Optional[int] = 3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # For self-supervised next-token or masked reconstruction:
            # Here labels are the input_ids themselves
            labels = input_ids.clone()
            
            self.optimizer.zero_grad()
            
            # Forward pass: attention mask passed correctly from dataset batch
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Flatten outputs and labels for CrossEntropyLoss
            vocab_size = outputs.size(-1)
            loss = self.criterion(
                outputs.view(-1, vocab_size),
                labels.view(-1)
            )
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if step % max(1, len(self.train_loader) // 5) == 0:
                logger.debug(f"Epoch {epoch+1} | Step {step}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = input_ids.clone()
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            vocab_size = outputs.size(-1)
            loss = self.criterion(
                outputs.view(-1, vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def fit(self, epochs: int) -> Dict[str, Any]:
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}
        
        logger.info("Starting training loop...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            history["train_loss"].append(train_loss)
            
            log_msg = f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}"
            
            if self.val_loader is not None:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                log_msg += f" | Val Loss: {val_loss:.4f}"
                
                # Checkpointing & Early Stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_loss, filename="best_model.pt")
                    logger.info(f"New best model saved with Val Loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                        logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            else:
                # Save checkpoint based on training loss if validation is not provided
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    self.save_checkpoint(epoch, train_loss, filename="best_model.pt")
                    
            logger.info(log_msg)
            
        # Save final model state
        self.save_checkpoint(epochs - 1, best_val_loss, filename="final_model.pt")
        logger.info("Training completed.")
        return history

    def save_checkpoint(self, epoch: int, loss: float, filename: str = "checkpoint.pt"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, filepath)
