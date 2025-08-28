#!/usr/bin/env python3
"""
FastAPI Server Script for Fraud Detection Pipeline
Serves the trained model via REST API.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fraud_pipeline.config import load_config
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start FastAPI server for fraud detection")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload on code changes"
    )
    parser.add_argument(
        "--workers", 
        type=int,
        default=1,
        help="Number of worker processes"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("serve")
    
    logger.info("🚀 Starting Fraud Detection API server...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Check if model exists
    model_path = Path(config.data.models_path) / "best_model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run 'make train' and 'make tune' first")
        return 1
    
    # Check if threshold exists
    threshold_path = Path(config.data.models_path) / "threshold.json"
    if not threshold_path.exists():
        logger.warning(f"Threshold file not found: {threshold_path}")
        logger.warning("Using default threshold from configuration")
    
    logger.info(f"Model found: {model_path}")
    logger.info(f"Threshold found: {threshold_path}")
    
    # Set environment variable for config path
    os.environ["CONFIG_PATH"] = args.config
    
    # Start server
    try:
        logger.info(f"Starting server on {args.host}:{args.port}")
        logger.info(f"API documentation available at: http://{args.host}:{args.port}/docs")
        logger.info(f"Health check available at: http://{args.host}:{args.port}/health")
        
        uvicorn.run(
            "fraud_pipeline.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
