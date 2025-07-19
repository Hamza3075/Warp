"""Model architecture definition

This file contains the neural network architecture for the SimpleAI system,
including a custom Transformer implementation.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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