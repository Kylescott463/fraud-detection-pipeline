#!/usr/bin/env python3
"""
Model Training Script for Fraud Detection Pipeline
Trains baseline models with MLflow integration.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.config import load_config
from fraud_pipeline.train import ModelTrainer
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force retrain even if models exist"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("train")
    
    logger.info("🚀 Starting model training...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Check if models already exist
    models_dir = Path(config.data.models_path)
    if not args.force:
        existing_models = list(models_dir.glob("*.joblib"))
        if existing_models:
            logger.info(f"Found {len(existing_models)} existing models:")
            for model_path in existing_models:
                logger.info(f"  - {model_path.name}")
            logger.info("Use --force to retrain all models")
            return 0
    
    try:
        # Train models
        results = trainer.train_models()
        
        # Print results table
        trainer.print_results_table(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
