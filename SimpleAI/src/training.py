"""Training loop and utilities

This file contains the training pipeline for NLP models, including training loop,
evaluation, and checkpoint management.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for NLP models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = False,
        task: str = "classification"  # Can be "classification", "generation", etc.
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for tracking
            task: Type of NLP task
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.task = task
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4
            )
        else:
            self.eval_loader = None
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            total_steps = len(self.train_loader) * num_epochs
            warmup_steps = int(0.1 * total_steps)
            self.scheduler = self._get_linear_schedule_with_warmup(
                self.optimizer, 
                warmup_steps, 
                total_steps
            )
        else:
            self.scheduler = scheduler
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize W&B if requested
        if use_wandb:
            import wandb
            wandb.init(project="nlp-system", config={
                "model_type": model.__class__.__name__,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "task": task,
            })
    
    def _get_linear_schedule_with_warmup(self, optimizer, warmup_steps, total_steps):
        """Create a schedule with linear warmup and linear decay"""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _compute_loss(self, batch):
        """Compute loss based on task type"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        if self.task == "classification":
            labels = batch["label"].to(self.device)
            logits = self.model(input_ids, attention_mask)
            
            # For classification, take the last token's representation
            sentence_repr = logits[:, 0, :]  # Use [CLS] token
            logits = self.model.output_projection(sentence_repr)
            
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        
        elif self.task == "generation":
            # Shift tokens for language modeling
            labels = input_ids.clone()
            labels = labels[:, 1:].contiguous()  # Remove first token
            
            logits = self.model(input_ids, attention_mask)
            logits = logits[:, :-1, :].contiguous()  # Remove last token prediction
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=self.model.pad_idx
            )
            return loss, logits
        
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def train(self):
        """Train the model"""
        logger.info(f"Starting training on {self.device}")
        self.model.to(self.device)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                loss, _ = self._compute_loss(batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                if self.use_wandb:
                    import wandb
                    wandb.log({"train_loss": loss.item()})
            
            avg_train_loss = train_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
            
            # Evaluation
            if self.eval_loader:
                val_loss, val_metrics = self.evaluate()
                logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
                
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"Epoch {epoch+1} - {metric_name}: {metric_value:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))
                    logger.info(f"Saved best model with val_loss {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}.pt"))
        
        logger.info("Training completed")
        
        # Save final model
        model_path = os.path.join(self.checkpoint_dir, "final_model")
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(model_path)
        else:
            torch.save(self.model.state_dict(), os.path.join(model_path, "model.pt"))
        
        return self.model
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                loss, logits = self._compute_loss(batch)
                val_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                if self.task == "classification":
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = batch["label"].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                
                if self.use_wandb:
                    import wandb
                    wandb.log({"val_loss": loss.item()})
        
        # Calculate average loss
        avg_val_loss = val_loss / len(self.eval_loader)
        
        # Calculate additional metrics
        metrics = {}
        if self.task == "classification" and len(all_preds) > 0:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["f1"] = f1_score(all_labels, all_preds, average='weighted')
            metrics["precision"] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics["recall"] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            if self.use_wandb:
                import wandb
                wandb.log(metrics)
        
        return avg_val_loss, metrics
    
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.num_epochs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint["epoch"]