import sys
import os
import json
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.model import NLPTransformer
from src.data import TextDataset
from src.training import Trainer
from src.inference import InferenceEngine
from src.api import APIServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy NLP model API")
    
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--model-path", type=str,
                        help="Path to trained model (overrides config)")
    parser.add_argument("--tokenizer-path", type=str,
                        help="Path to tokenizer (overrides config)")
    parser.add_argument("--host", type=str,
                        help="Host to bind server to (overrides config)")
    parser.add_argument("--port", type=int,
                        help="Port to bind server to (overrides config)")
    parser.add_argument("--batch-size", type=int,
                        help="Inference batch size (overrides config)")
    parser.add_argument("--cache-size", type=int,
                        help="Cache size (overrides config)")
    
    return parser.parse_args()

def main():
    """Main deployment function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Override config with command line arguments
    inference_config = config["inference"]
    api_config = config["api"]
    
    if args.model_path:
        inference_config["model_path"] = args.model_path
    if args.tokenizer_path:
        inference_config["tokenizer_path"] = args.tokenizer_path
    if args.batch_size:
        inference_config["batch_size"] = args.batch_size
    if args.cache_size:
        inference_config["cache_size"] = args.cache_size
    if args.host:
        api_config["host"] = args.host
    if args.port:
        api_config["port"] = args.port
    
    # Validate and normalize model path
    try:
        model_path = inference_config["model_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", model_path)
            )
        
        # Verify the model path exists and points to a file
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            logger.error("Please run training first or provide a valid model path")
            return
        
        if not os.path.isfile(model_path):
            logger.error(f"Model path is not a file: {model_path}")
            logger.error("Please provide a valid model file path")
            return
            
        inference_config["model_path"] = model_path
        
    except Exception as e:
        logger.error(f"Error validating model path: {str(e)}")
        return
    
    # Create inference engine
    logger.info("Creating inference engine...")
    inference_engine = InferenceEngine(
        model_path=inference_config["model_path"],
        tokenizer_path=inference_config["tokenizer_path"],
        cache_size=inference_config["cache_size"],
        batch_size=inference_config["batch_size"],
        max_length=inference_config.get("max_length", 512)
    )
    
    # Create and start API server
    logger.info(f"Starting API server on {api_config['host']}:{api_config['port']}...")
    api_server = APIServer(
        inference_engine=inference_engine,
        host=api_config["host"],
        port=api_config["port"]
    )
    
    api_server.start()

if __name__ == "__main__":
    main()