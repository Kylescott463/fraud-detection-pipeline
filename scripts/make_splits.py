#!/usr/bin/env python3
"""
Data Splitting Script for Fraud Detection Pipeline
Preprocesses data and creates train/test splits with imbalanced handling.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fraud_pipeline.config import load_config
from fraud_pipeline.preprocess import DataPreprocessor
from fraud_pipeline.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create train/test splits for Credit Card Fraud dataset")
    parser.add_argument(
        "--config", 
        default="fraud_pipeline/configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force regenerate splits even if they exist"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("make_splits")
    
    logger.info("🚀 Starting data splitting process...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Check if splits already exist
    processed_dir = Path(config.data.processed_path)
    required_files = ["X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet"]
    
    if not args.force and all((processed_dir / file).exists() for file in required_files):
        logger.info("Data splits already exist. Use --force to regenerate.")
        
        # Load and display existing split info
        try:
            y_train = pd.read_parquet(processed_dir / "y_train.parquet").iloc[:, 0]
            y_test = pd.read_parquet(processed_dir / "y_test.parquet").iloc[:, 0]
            
            split_info = preprocessor.get_split_info(y_train, y_test)
            
            print("\n" + "="*60)
            print("📊 EXISTING DATA SPLITS")
            print("="*60)
            print(f"🎯 Sampling Strategy: {config.sampling.strategy}")
            print(f"📈 Features: {len(pd.read_parquet(processed_dir / 'X_train.parquet').columns)}")
            
            print(f"\n📚 TRAIN SET:")
            print(f"   Total samples: {split_info['train']['total_samples']:,}")
            print(f"   Legitimate: {split_info['train']['legitimate_samples']:,}")
            print(f"   Fraudulent: {split_info['train']['fraudulent_samples']:,}")
            print(f"   Fraud rate: {split_info['train']['fraud_rate']:.4f}%")
            
            print(f"\n🧪 TEST SET:")
            print(f"   Total samples: {split_info['test']['total_samples']:,}")
            print(f"   Legitimate: {split_info['test']['legitimate_samples']:,}")
            print(f"   Fraudulent: {split_info['test']['fraudulent_samples']:,}")
            print(f"   Fraud rate: {split_info['test']['fraud_rate']:.4f}%")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error reading existing splits: {e}")
            return 1
    
    try:
        # Run preprocessing pipeline
        results = preprocessor.preprocess_data()
        
        # Print results
        print("\n" + "="*60)
        print("📊 DATA SPLITTING RESULTS")
        print("="*60)
        print(f"🎯 Sampling Strategy: {results['sampling_strategy']}")
        print(f"📈 Features: {len(results['feature_names'])}")
        print(f"🔧 Features: {', '.join(results['feature_names'])}")
        
        split_info = results['split_info']
        
        print(f"\n📚 TRAIN SET:")
        print(f"   Total samples: {split_info['train']['total_samples']:,}")
        print(f"   Legitimate: {split_info['train']['legitimate_samples']:,}")
        print(f"   Fraudulent: {split_info['train']['fraudulent_samples']:,}")
        print(f"   Fraud rate: {split_info['train']['fraud_rate']:.4f}%")
        
        print(f"\n🧪 TEST SET:")
        print(f"   Total samples: {split_info['test']['total_samples']:,}")
        print(f"   Legitimate: {split_info['test']['legitimate_samples']:,}")
        print(f"   Fraudulent: {split_info['test']['fraudulent_samples']:,}")
        print(f"   Fraud rate: {split_info['test']['fraud_rate']:.4f}%")
        
        print(f"\n💾 SAVED ARTIFACTS:")
        print(f"   📄 X_train.parquet: {processed_dir / 'X_train.parquet'}")
        print(f"   📄 X_test.parquet: {processed_dir / 'X_test.parquet'}")
        print(f"   📄 y_train.parquet: {processed_dir / 'y_train.parquet'}")
        print(f"   📄 y_test.parquet: {processed_dir / 'y_test.parquet'}")
        print(f"   🔧 StandardScaler: {processed_dir / 'transformers' / 'standard_scaler.joblib'}")
        
        if preprocessor.sampler is not None:
            print(f"   🔧 Sampler: {processed_dir / 'transformers' / 'sampler.joblib'}")
        
        print("\n✅ Data splitting completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in data splitting: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
