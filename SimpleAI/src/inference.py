"""Inference engine with caching

This file contains the inference engine for NLP models, including caching
and batching mechanisms for optimized performance.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


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
            from model import NLPTransformer
            self.model = NLPTransformer.from_pretrained(model_path)
        else:
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