#!/usr/bin/env python3
"""
Data Download Script for Fraud Detection Pipeline
Downloads and validates the Kaggle Credit Card Fraud dataset.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from fraud_pipeline.config import load_config
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download Credit Card Fraud dataset")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if files exist"
    )
    return parser.parse_args()


def check_kaggle_credentials():
    """Check if Kaggle credentials are available."""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if kaggle_username and kaggle_key:
        return True
    
    # Check if kaggle.json exists
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    
    return False


def download_from_kaggle(output_dir: Path, logger):
    """Download dataset from Kaggle API."""
    logger.info("Downloading dataset from Kaggle...")
    
    try:
        # Download the dataset
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "mlg-ulb/creditcardfraud",
            "-p", str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"Kaggle download failed: {result.stderr}")
            return False
        
        logger.info("Dataset downloaded successfully")
        
        # Extract the zip file
        zip_file = output_dir / "creditcardfraud.zip"
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Remove the zip file
            zip_file.unlink()
            logger.info("Dataset extracted successfully")
            return True
        else:
            logger.error("Downloaded zip file not found")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Kaggle download timed out")
        return False
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {e}")
        return False


def copy_local_file(source_path: str, output_dir: Path, logger):
    """Copy local file to output directory."""
    source = Path(source_path)
    
    if not source.exists():
        logger.error(f"Source file not found: {source_path}")
        return False
    
    logger.info(f"Copying local file: {source_path}")
    
    try:
        shutil.copy2(source, output_dir / "creditcard.csv")
        logger.info("Local file copied successfully")
        return True
    except Exception as e:
        logger.error(f"Error copying local file: {e}")
        return False


def validate_dataset(csv_path: Path, logger):
    """Validate the dataset structure and content."""
    logger.info("Validating dataset...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check expected columns
        expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False, None
        
        # Check data types
        if df['Class'].dtype not in ['int64', 'int32']:
            logger.error("Class column should be integer type")
            return False, None
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Check class distribution
        class_counts = df['Class'].value_counts().sort_index()
        total_rows = len(df)
        fraud_rate = (class_counts.get(1, 0) / total_rows) * 100
        
        logger.info(f"Dataset validation successful:")
        logger.info(f"  - Total rows: {total_rows:,}")
        logger.info(f"  - Columns: {len(df.columns)}")
        logger.info(f"  - Class distribution: {class_counts.to_dict()}")
        logger.info(f"  - Fraud rate: {fraud_rate:.4f}%")
        
        return True, df
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False, None


def save_parquet(df: pd.DataFrame, output_path: Path, logger):
    """Save dataset as Parquet file."""
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"Parquet file saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving Parquet file: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("download_data")
    
    logger.info("🚀 Starting data download process...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(config.data.raw_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "creditcard.csv"
    parquet_path = output_dir / "creditcard.parquet"
    
    # Check if files already exist
    if not args.force and csv_path.exists() and parquet_path.exists():
        logger.info("Dataset files already exist. Use --force to re-download.")
        
        # Validate existing files
        success, df = validate_dataset(csv_path, logger)
        if success:
            logger.info("✅ Existing dataset is valid")
            return 0
        else:
            logger.error("❌ Existing dataset is invalid")
            return 1
    
    # Determine data source
    data_path = os.getenv("DATA_PATH")
    
    if data_path and Path(data_path).exists():
        # Copy local file
        logger.info("Using local DATA_PATH")
        if not copy_local_file(data_path, output_dir, logger):
            return 1
    elif check_kaggle_credentials():
        # Download from Kaggle
        logger.info("Using Kaggle API")
        if not download_from_kaggle(output_dir, logger):
            return 1
    else:
        logger.error("No data source available:")
        logger.error("  - DATA_PATH not set or file not found")
        logger.error("  - Kaggle credentials not available")
        logger.error("Please set DATA_PATH or configure Kaggle credentials")
        return 1
    
    # Validate the dataset
    success, df = validate_dataset(csv_path, logger)
    if not success:
        logger.error("❌ Dataset validation failed")
        return 1
    
    # Save Parquet version
    if not save_parquet(df, parquet_path, logger):
        return 1
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    print(f"📁 Location: {output_dir}")
    print(f"📄 CSV file: {csv_path}")
    print(f"📊 Parquet file: {parquet_path}")
    print(f"📈 Total rows: {len(df):,}")
    print(f"📋 Columns: {len(df.columns)}")
    
    # Class distribution
    class_counts = df['Class'].value_counts().sort_index()
    fraud_rate = (class_counts.get(1, 0) / len(df)) * 100
    
    print(f"\n🎯 CLASS DISTRIBUTION:")
    print(f"   Legitimate (0): {class_counts.get(0, 0):,} ({100-fraud_rate:.2f}%)")
    print(f"   Fraudulent (1): {class_counts.get(1, 0):,} ({fraud_rate:.2f}%)")
    print(f"   Imbalance ratio: 1:{class_counts.get(0, 0)/class_counts.get(1, 0):.1f}")
    
    print("\n✅ Data download completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
