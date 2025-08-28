#!/usr/bin/env python3
"""
Batch Scoring Script for Fraud Detection Pipeline
Scores a batch of transactions from CSV and saves results to Parquet.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from fraud_pipeline.config import load_config
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Score batch of transactions")
    parser.add_argument(
        "--input", 
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output", 
        required=True,
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", 
        default="best_model.joblib",
        help="Model file name (relative to models directory)"
    )
    parser.add_argument(
        "--threshold", 
        type=float,
        help="Custom threshold (overrides saved threshold)"
    )
    return parser.parse_args()


def load_model_and_threshold(config, model_name: str, custom_threshold: float = None):
    """Load model and threshold.
    
    Args:
        config: Configuration object
        model_name: Name of model file
        custom_threshold: Custom threshold value
        
    Returns:
        Tuple of (model, threshold, scaler)
    """
    # Load model
    model_path = Path(config.data.models_path) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = Path(config.data.processed_path) / "transformers" / "standard_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    # Load threshold
    if custom_threshold is not None:
        threshold = custom_threshold
    else:
        threshold_path = Path(config.data.models_path) / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data["threshold"]
        else:
            threshold = config.serving.threshold
    
    return model, threshold, scaler


def validate_input_data(df: pd.DataFrame, logger=None) -> None:
    """Validate input data format.
    
    Args:
        df: Input DataFrame
        logger: Optional logger for info messages
        
    Raises:
        ValueError: If data format is invalid
    """
    required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for non-numeric values
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    
    # Check for negative amounts
    if (df["Amount"] < 0).any():
        raise ValueError("Amount values must be non-negative")
    
    # Check for negative time
    if (df["Time"] < 0).any():
        raise ValueError("Time values must be non-negative")
    
    # Filter out non-required columns (like 'Class')
    extra_columns = [col for col in df.columns if col not in required_columns]
    if extra_columns and logger:
        logger.info(f"Filtering out extra columns: {extra_columns}")
        df.drop(columns=extra_columns, inplace=True)


def score_batch(model, threshold: float, df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    """Score a batch of transactions.
    
    Args:
        model: Trained model
        threshold: Decision threshold
        df: Input DataFrame
        scaler: Optional scaler for preprocessing
        
    Returns:
        DataFrame with predictions
    """
    # Apply preprocessing to match the training data format
    if scaler is not None:
        # Scale Amount and Time features separately (as done in Stage 4)
        amount_scaled = scaler.transform(df[['Amount']].values.reshape(-1, 1))
        df_processed = df.copy()
        df_processed['Amount_scaled'] = amount_scaled.flatten()
        
        # For Time, we need to check if it was included in training
        if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
            time_scaled = scaler.transform(df[['Time']].values.reshape(-1, 1))
            df_processed['Time_scaled'] = time_scaled.flatten()
        
        # Drop original Amount and Time columns
        df_processed = df_processed.drop(['Amount', 'Time'], axis=1)
    else:
        # If no scaler, create dummy scaled columns
        df_processed = df.copy()
        df_processed['Amount_scaled'] = df_processed['Amount']
        if hasattr(model.named_steps['scaler'], 'feature_names_in_') and 'Time_scaled' in model.named_steps['scaler'].feature_names_in_:
            df_processed['Time_scaled'] = df_processed['Time']
        df_processed = df_processed.drop(['Amount', 'Time'], axis=1)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(df_processed)[:, 1]
    
    # Make decisions based on threshold
    decisions = (probabilities >= threshold).astype(int)
    
    # Create results DataFrame
    results = df.copy()
    results["fraud_probability"] = probabilities
    results["fraud_decision"] = decisions
    results["threshold"] = threshold
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("score_batch")
    
    logger.info("📊 Starting batch scoring...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Load input data
    try:
        logger.info(f"Loading input data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} transactions with {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        return 1
    
    # Validate input data
    try:
        validate_input_data(df, logger)
        logger.info("Input data validation passed")
    except ValueError as e:
        logger.error(f"Input data validation failed: {e}")
        return 1
    
    # Load model and threshold
    try:
        model, threshold, scaler = load_model_and_threshold(config, args.model, args.threshold)
        logger.info(f"Model loaded: {args.model}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Scaler loaded: {scaler is not None}")
    except Exception as e:
        logger.error(f"Error loading model and threshold: {e}")
        return 1
    
    # Score batch
    try:
        logger.info("Scoring transactions...")
        results = score_batch(model, threshold, df, scaler)
        logger.info("Scoring completed")
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        return 1
    
    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_parquet(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return 1
    
    # Print summary
    total_transactions = len(results)
    fraud_count = results["fraud_decision"].sum()
    fraud_rate = fraud_count / total_transactions
    avg_probability = results["fraud_probability"].mean()
    
    print("\n" + "="*60)
    print("📊 BATCH SCORING SUMMARY")
    print("="*60)
    print(f"Total Transactions: {total_transactions:,}")
    print(f"Fraud Detected: {fraud_count:,}")
    print(f"Fraud Rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    print(f"Average Probability: {avg_probability:.4f}")
    print(f"Threshold Used: {threshold:.4f}")
    print(f"Results saved to: {output_path}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
