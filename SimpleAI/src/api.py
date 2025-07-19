"""FastAPI server implementation

This file contains the API server implementation for the SimpleAI system,
including endpoints for model inference and health checks.
"""

import logging
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.inference import InferenceEngine

logger = logging.getLogger(__name__)


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


def create_api_server(model_path: str, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> APIServer:
    """
    Create and configure an API server
    
    Args:
        model_path: Path to the model to use for inference
        host: Host to bind server to
        port: Port to bind server to
        **kwargs: Additional arguments for the inference engine
        
    Returns:
        Configured API server
    """
    # Create inference engine
    inference_engine = InferenceEngine(
        model_path=model_path,
        **kwargs
    )
    
    # Create API server
    api_server = APIServer(
        inference_engine=inference_engine,
        host=host,
        port=port
    )
    
    return api_server