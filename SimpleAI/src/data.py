"""Data loading and preprocessing

This file contains utilities for loading and preprocessing text data for NLP tasks.
"""

import json
import csv
from typing import List, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
        
        # Define label mapping for text labels
        label_mapping = {
            "positive": 1,
            "negative": 0,
            "neutral": 2,
            "true": 1,
            "false": 0
        }
        
        # Load data from file (CSV or JSON format)
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find text and label columns
                text_col = header.index("text") if "text" in header else 0
                label_col = header.index("label") if "label" in header and task == "classification" else None
                
                for row in reader:
                    # Check if row has enough columns
                    if len(row) <= text_col:
                        continue
                    
                    texts.append(row[text_col])
                    if label_col is not None and len(row) > label_col:
                        label_value = row[label_col].lower()
                        if label_value in label_mapping:
                            labels.append(label_mapping[label_value])
                        else:
                            try:
                                labels.append(int(label_value))
                            except ValueError:
                                raise ValueError(f"Invalid label value: {label_value}. Must be one of {list(label_mapping.keys())} or an integer.")
                        
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for item in data:
                    if isinstance(item, dict):
                        texts.append(item.get("text", ""))
                        if task == "classification" and "label" in item:
                            label_value = str(item["label"]).lower()
                            if label_value in label_mapping:
                                labels.append(label_mapping[label_value])
                            else:
                                try:
                                    labels.append(int(label_value))
                                except ValueError:
                                    raise ValueError(f"Invalid label value: {label_value}. Must be one of {list(label_mapping.keys())} or an integer.")
                    else:
                        texts.append(item)
        
        return cls(texts, labels, tokenizer, max_length)


def create_dataloaders(train_dataset, eval_dataset=None, batch_size=32, num_workers=4):
    """
    Create DataLoader objects for training and evaluation
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training
        eval_loader: DataLoader for evaluation (or None)
    """
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    eval_loader = None
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    return train_loader, eval_loader