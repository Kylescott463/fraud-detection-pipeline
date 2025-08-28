#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Fraud Detection Pipeline
Tunes hyperparameters for the best models using Optuna or RandomizedSearchCV.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.config import load_config
from fraud_pipeline.tune import HyperparameterTuner
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for fraud detection models")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--trials", 
        type=int,
        default=50,
        help="Number of trials for hyperparameter optimization"
    )
    parser.add_argument(
        "--models", 
        nargs="+",
        default=["random_forest", "logistic_regression"],
        help="Models to tune (default: random_forest logistic_regression)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-tuning even if best_model.joblib exists"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("tune")
    
    logger.info("🎯 Starting hyperparameter tuning...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Check if best model already exists
    best_model_path = Path(config.data.models_path) / "best_model.joblib"
    if best_model_path.exists() and not args.force:
        logger.info("Best model already exists. Use --force to re-tune.")
        return 0
    
    # Initialize tuner
    tuner = HyperparameterTuner(config)
    
    # Validate models
    valid_models = ["logistic_regression", "random_forest"]
    models_to_tune = [model for model in args.models if model in valid_models]
    
    if not models_to_tune:
        logger.error(f"No valid models specified. Valid models: {valid_models}")
        return 1
    
    logger.info(f"Tuning models: {models_to_tune}")
    logger.info(f"Number of trials: {args.trials}")
    
    try:
        # Tune models
        results = tuner.tune_models(models_to_tune, n_trials=args.trials)
        
        if not results:
            logger.error("No models were successfully tuned")
            return 1
        
        # Select best model
        best_model_name, best_model_results = tuner.select_best_model(results)
        
        # Save best model
        tuner.save_best_model(best_model_name, best_model_results)
        
        # Print results
        tuner.print_results(results, best_model_name)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during tuning: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
