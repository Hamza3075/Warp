"""
Production-Ready NLP System
===========================
This implementation provides a comprehensive NLP system with:
- Model architecture (Transformer-based)
- Data preprocessing pipeline
- Training & evaluation loops
- Inference engine with caching
- API server implementation
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import time
import pickle
import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import wandb  # For experiment tracking

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

# ====================== MODEL ARCHITECTURE ======================

class TransformerEncoderLayer(nn.Module):
    """Custom implementation of a Transformer Encoder Layer"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.gelu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        return src


class NLPTransformer(nn.Module):
    """Transformer model for NLP tasks"""
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 768, 
        nhead: int = 12, 
        num_encoder_layers: int = 12,
        dim_feedforward: int = 3072, 
        dropout: float = 0.1,
        max_seq_length: int = 512,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token embeddings and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head (can be replaced for different tasks)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model
        
        Args:
            input_ids: Tensor of token indices [batch_size, seq_len]
            attention_mask: Mask to avoid attention on padding tokens [batch_size, seq_len]
        
        Returns:
            output: Tensor with predictions [batch_size, seq_len, vocab_size]
        """
        seq_length = input_ids.size(1)
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Create embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).float()
        
        # Create key padding mask for attention layers
        key_padding_mask = (attention_mask == 0)
        
        # Pass through encoder layers
        hidden_states = embeddings.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, src_key_padding_mask=key_padding_mask)
            
        # Convert back to [batch_size, seq_len, d_model]
        hidden_states = hidden_states.transpose(0, 1)
        
        # Get token predictions
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def save_pretrained(self, path: str):
        """Save model weights and configuration to a directory"""
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save model config
        config = {
            "vocab_size": self.token_embedding.weight.size(0),
            "d_model": self.d_model,
            "nhead": self.encoder_layers[0].self_attn.num_heads,
            "num_encoder_layers": len(self.encoder_layers),
            "dim_feedforward": self.encoder_layers[0].linear1.out_features,
            "dropout": self.dropout.p,
            "max_seq_length": self.position_embedding.weight.size(0),
            "pad_idx": self.pad_idx
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load model weights and configuration from a directory"""
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        
        return model

# ====================== DATA PREPROCESSING ======================

class TextDataset(Dataset):
    """Dataset for text data"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of text samples
            labels: Optional list of labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert to 1D tensors
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Return encoded text and label if available
        if self.labels is not None:
            label = self.labels[idx]
            return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer=None, max_length: int = 512, task: str = "classification"):
        """Load dataset from a file"""
        texts = []
        labels = [] if task == "classification" else None
        
        # Load data from file (CSV or JSON format)
        if file_path.endswith('.csv'):
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find text and label columns
                text_col = header.index("text") if "text" in header else 0
                label_col = header.index("label") if "label" in header and task == "classification" else None
                
                for row in reader:
                    texts.append(row[text_col])
                    if label_col is not None:
                        labels.append(int(row[label_col]))
                        
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for item in data:
                    if isinstance(item, dict):
                        texts.append(item.get("text", ""))
                        if task == "classification" and "label" in item:
                            labels.append(int(item["label"]))
                    else:
                        texts.append(item)
        
        return cls(texts, labels, tokenizer, max_length)


# ====================== TRAINING PIPELINE ======================

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

# ====================== INFERENCE ENGINE ======================

class InferenceEngine:
    """Inference engine with caching and batching"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_size: int = 1000,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer (defaults to model path if None)
            device: Device to use for inference
            cache_size: Maximum number of entries in cache
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        # Load model
        if os.path.isfile(os.path.join(model_path, "config.json")):
            self.model = NLPTransformer.from_pretrained(model_path)
        else:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        self.device = device
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize cache
        self.cache = {}
        self.cache_order = []  # To track LRU entries
    
    def _get_from_cache(self, text):
        """Get result from cache if available"""
        if text in self.cache:
            # Move entry to the end (most recently used)
            self.cache_order.remove(text)
            self.cache_order.append(text)
            return self.cache[text]
        return None
    
    def _add_to_cache(self, text, result):
        """Add result to cache"""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used entry
            oldest_text = self.cache_order.pop(0)
            del self.cache[oldest_text]
        
        self.cache[text] = result
        self.cache_order.append(text)
    
    def predict(self, texts: Union[str, List[str]], **kwargs) -> Union[Dict, List[Dict]]:
        """
        Make predictions for the given texts
        
        Args:
            texts: Single text or list of texts
            kwargs: Additional arguments for prediction
        
        Returns:
            Dictionary or list of dictionaries with predictions
        """
        # Handle single text
        if isinstance(texts, str):
            return self.predict([texts], **kwargs)[0]
        
        # Check cache first
        results = [self._get_from_cache(text) for text in texts]
        
        # Find texts that are not in cache
        uncached_indices = [i for i, r in enumerate(results) if r is None]
        
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            
            # Process in batches
            all_batch_results = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[i:i+self.batch_size]
                batch_results = self._predict_batch(batch_texts, **kwargs)
                all_batch_results.extend(batch_results)
            
            # Update cache and results
            for i, result in zip(uncached_indices, all_batch_results):
                text = texts[i]
                self._add_to_cache(text, result)
                results[i] = result
        
        return results
    
    def _predict_batch(self, texts: List[str], **kwargs) -> List[Dict]:
        """Make predictions for a batch of texts"""
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs based on model type
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Convert to numpy for processing
        logits_np = logits.cpu().numpy()
        
        # Format results (will depend on the task)
        results = []
        for i, logit in enumerate(logits_np):
            # For classification
            if len(logit.shape) == 1 or logit.shape[0] == 1:
                probs = softmax(logit)
                pred_class = np.argmax(probs)
                
                result = {
                    "prediction": int(pred_class),
                    "probabilities": {str(j): float(p) for j, p in enumerate(probs)}
                }
            # For token prediction / generation
            else:
                # Get top predictions for each position
                top_tokens = np.argsort(-logit, axis=1)[:, :5]  # Top 5 for each position
                result = {
                    "tokens": self.tokenizer.decode(np.argmax(logit, axis=1)),
                    "top_tokens": [[self.tokenizer.decode([token]) for token in position] 
                                  for position in top_tokens]
                }
            
            results.append(result)
        
        return results

def softmax(x):
    """Compute softmax values for each set of scores in x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ====================== API SERVER ======================

class APIServer:
    """FastAPI server for model inference"""
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize API server
        
        Args:
            inference_engine: Inference engine to use
            host: Host to bind server to
            port: Port to bind server to
        """
        self.inference_engine = inference_engine
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="NLP API",
            description="API for NLP tasks",
            version="1.0.0"
        )
        
        # Define request models
        class PredictionRequest(BaseModel):
            texts: Union[str, List[str]]
            options: Optional[Dict] = {}
        
        # Define routes
        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            try:
                result = self.inference_engine.predict(request.texts, **request.options)
                return {"success": True, "predictions": result}
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}
    
    def start(self):
        """Start the API server"""
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

# ====================== MAIN ======================

def main():
    """Example usage of the system"""
    # Define configuration
    config = {
        "model": {
            "vocab_size": 30522,  # BERT vocab size
            "d_model": 768,
            "nhead": 12,
            "num_encoder_layers": 12,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "max_seq_length": 512,
            "pad_idx": 0
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "train_file": "data/train.csv",
            "eval_file": "data/eval.csv",
            "checkpoint_dir": "./checkpoints",
            "use_wandb": False
        },
        "inference": {
            "model_path": "./checkpoints/final_model",
            "cache_size": 1000,
            "batch_size": 32
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000
        }
    }
    
    # Create model
    model = NLPTransformer(**config["model"])
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = TextDataset.from_file(
        config["training"]["train_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_seq_length"],
        task="classification"
    )
    
    eval_dataset = TextDataset.from_file(
        config["training"]["eval_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_seq_length"],
        task="classification"
    )
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        use_wandb=config["training"]["use_wandb"],
        task="classification"
    )
    
    # Train model
    trainer.train()
    
    # Create inference engine
    inference_engine = InferenceEngine(
        model_path=config["inference"]["model_path"],
        tokenizer_path=None,  # Use same path as model
        cache_size=config["inference"]["cache_size"],
        batch_size=config["inference"]["batch_size"]
    )
    
    # Create and start API server
    api_server = APIServer(
        inference_engine=inference_engine,
        host=config["api"]["host"],
        port=config["api"]["port"]
    )
    api_server.start()


if __name__ == "__main__":
    main()