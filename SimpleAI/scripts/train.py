#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the NLP model.
This script loads the configuration from config.json and trains the model.
"""

import os
import sys
import json
import argparse
import logging
from transformers import AutoTokenizer

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import NLPTransformer
from src.data import TextDataset
from src.training import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train NLP model")
    
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--train-file", type=str,
                        help="Path to training data file (overrides config)")
    parser.add_argument("--eval-file", type=str,
                        help="Path to evaluation data file (overrides config)")
    parser.add_argument("--checkpoint-dir", type=str,
                        help="Directory to save checkpoints (overrides config)")
    parser.add_argument("--batch-size", type=int,
                        help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate (overrides config)")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for tracking (overrides config)")
    parser.add_argument("--task", type=str, choices=["classification", "generation"],
                        help="Task type (overrides config)")
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    
    if args.train_file:
        training_config["train_file"] = args.train_file
    if args.eval_file:
        training_config["eval_file"] = args.eval_file
    if args.checkpoint_dir:
        training_config["checkpoint_dir"] = args.checkpoint_dir
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    if args.epochs:
        training_config["num_epochs"] = args.epochs
    if args.use_wandb:
        training_config["use_wandb"] = args.use_wandb
    if args.task:
        training_config["task"] = args.task
    if args.resume:
        training_config["resume_from"] = args.resume
        
    # Check if training data file exists
    train_file = training_config.get("train_file")
    if train_file and not os.path.exists(train_file):
        logger.error(f"Training data file not found: {train_file}")
        logger.info("Please make sure the data file exists or update the path in config.json")
        return
    
    # Create model
    logger.info("Creating model...")
    model = NLPTransformer(**config["model"])
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare datasets
    logger.info("Loading datasets...")
    train_dataset = TextDataset.from_file(
        training_config["train_file"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_seq_length"],
        task=training_config["task"]
    )
    
    eval_dataset = None
    if training_config["eval_file"]:
        eval_dataset = TextDataset.from_file(
            training_config["eval_file"],
            tokenizer=tokenizer,
            max_length=config["model"]["max_seq_length"],
            task=training_config["task"]
        )
    
    logger.info(f"Loaded {len(train_dataset)} training samples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Create checkpoint directory
    os.makedirs(training_config["checkpoint_dir"], exist_ok=True)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        num_epochs=training_config["num_epochs"],
        checkpoint_dir=training_config["checkpoint_dir"],
        use_wandb=training_config["use_wandb"],
        task=training_config["task"]
    )
    
    # Resume training if requested
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()