import unittest
import torch
import os
import tempfile
import json
import shutil

# Import your model components
from src.model import NLPTransformer
from src.data import TextDataset
from src.inference import InferenceEngine

class TestNLPModel(unittest.TestCase):
    """Test cases for the NLP model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create a small model for testing
        self.model_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "max_seq_length": 64,
            "pad_idx": 0
        }
        
        self.model = NLPTransformer(**self.model_config)
        
        # Create sample input
        self.batch_size = 2
        self.seq_len = 10
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)
        
    def tearDown(self):
        """Tear down test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Run forward pass
        outputs = self.model(self.input_ids, self.attention_mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.model_config["vocab_size"])
        self.assertEqual(outputs.shape, expected_shape)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        model_path = os.path.join(self.test_dir, "test_model")
        
        # Save model
        self.model.save_pretrained(model_path)
        
        # Check that files exist
        self.assertTrue(os.path.exists(os.path.join(model_path, "model.pt")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "config.json")))
        
        # Load model
        loaded_model = NLPTransformer.from_pretrained(model_path)
        
        # Run inference with both models
        with torch.no_grad():
            original_output = self.model(self.input_ids, self.attention_mask)
            loaded_output = loaded_model(self.input_ids, self.attention_mask)
        
        # Check that outputs are the same
        torch.testing.assert_close(original_output, loaded_output)
    
    def test_dataset(self):
        """Test dataset functionality"""
        # Create sample data
        texts = ["This is a test", "Another test sentence"]
        labels = [0, 1]
        
        # Create dataset with mock tokenizer
        class MockTokenizer:
            def __call__(self, text, truncation=None, max_length=None, padding=None, return_tensors=None):
                # Mock tokenization
                # Just return random tensors of the right shape
                if isinstance(text, list):
                    input_ids = [torch.randint(0, 1000, (min(len(t.split()), max_length),)) for t in text]
                    attention_mask = [torch.ones_like(ids) for ids in input_ids]
                    
                    # Pad to max_length
                    padded_input_ids = []
                    padded_attention_mask = []
                    for ids, mask in zip(input_ids, attention_mask):
                        if len(ids) < max_length:
                            padding_length = max_length - len(ids)
                            padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
                            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
                        else:
                            padded_ids = ids[:max_length]
                            padded_mask = mask[:max_length]
                        padded_input_ids.append(padded_ids)
                        padded_attention_mask.append(padded_mask)
                    
                    # Convert to tensors
                    input_ids = torch.stack(padded_input_ids)
                    attention_mask = torch.stack(padded_attention_mask)
                else:
                    # Single text
                    input_ids = torch.randint(0, 1000, (min(len(text.split()), max_length),))
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Pad to max_length
                    if len(input_ids) < max_length:
                        padding_length = max_length - len(input_ids)
                        input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
                        attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                    else:
                        input_ids = input_ids[:max_length]
                        attention_mask = attention_mask[:max_length]
                    
                    # Add batch dimension if requested
                    if return_tensors == "pt":
                        input_ids = input_ids.unsqueeze(0)
                        attention_mask = attention_mask.unsqueeze(0)
                
                return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        dataset = TextDataset(texts, labels, tokenizer=MockTokenizer(), max_length=64)
        
        # Check dataset length
        self.assertEqual(len(dataset), len(texts))
        
        # Check dataset item
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("label", item)
        self.assertEqual(item["label"], labels[0])
    
    def test_inference_engine(self):
        """Test inference engine"""
        model_path = os.path.join(self.test_dir, "test_model")
        
        # Save model
        self.model.save_pretrained(model_path)
        
        # Mock tokenizer for inference engine
        class MockTokenizer:
            def __init__(self):
                pass
                
            def __call__(self, text, truncation=None, max_length=None, padding=None, return_tensors=None):
                # Mock tokenization
                if isinstance(text, list):
                    batch_size = len(text)
                else:
                    batch_size = 1
                    
                input_ids = torch.randint(0, 1000, (batch_size, max_length))
                attention_mask = torch.ones_like(input_ids)
                
                return {"input_ids": input_ids, "attention_mask": attention_mask}
                
            def decode(self, token_ids):
                # Mock decoding
                if isinstance(token_ids, list):
                    return ["decoded text" for _ in token_ids]
                else:
                    return "decoded text"
                    
            @classmethod
            def from_pretrained(cls, path):
                return cls()
        
        # Create a simple inference engine
        # We'll patch the InferenceEngine to use our mock tokenizer
        original_tokenizer = InferenceEngine.tokenizer
        InferenceEngine.tokenizer = MockTokenizer.from_pretrained("test")
        
        engine = InferenceEngine(
            model_path=model_path,
            cache_size=10,
            batch_size=2
        )
        
        # Test single prediction
        result = engine.predict("This is a test")
        self.assertIsInstance(result, dict)
        
        # Test batch prediction
        results = engine.predict(["Test 1", "Test 2"])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        # Test cache
        engine.predict("Repeated text")
        result = engine.predict("Repeated text")  # Should come from cache
        self.assertIsInstance(result, dict)
        
        # Restore original tokenizer
        InferenceEngine.tokenizer = original_tokenizer

if __name__ == "__main__":
    unittest.main()