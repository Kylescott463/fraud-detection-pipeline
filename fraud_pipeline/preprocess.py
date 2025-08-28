"""Data preprocessing module for fraud detection pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

from .utils.logger import setup_logger


class DataPreprocessor:
    """Data preprocessor for fraud detection pipeline."""
    
    def __init__(self, config):
        """Initialize data preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("data_preprocessor")
        self.processed_dir = Path(config.data.processed_path)
        self.transformers_dir = self.processed_dir / "transformers"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.transformers_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.sampler = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from parquet file.
        
        Returns:
            Loaded DataFrame
        """
        parquet_path = Path(self.config.data.raw_path).parent / "creditcard.parquet"
        if parquet_path.exists():
            self.logger.info(f"Loading data from {parquet_path}")
            return pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        self.logger.info("Preparing features and target...")
        
        # Extract target
        target = df['Class'].copy()
        
        # Prepare features
        feature_cols = []
        
        # Add V1-V28 features
        v_features = [f'V{i}' for i in range(1, 29)]
        feature_cols.extend(v_features)
        
        # Add scaled Amount
        amount_scaled = self.scaler.fit_transform(df[['Amount']].values.reshape(-1, 1))
        features_df = df[v_features].copy()
        features_df['Amount_scaled'] = amount_scaled.flatten()
        feature_cols.append('Amount_scaled')
        
        # Optionally add scaled Time
        if self.config.sampling.include_time:
            time_scaled = self.scaler.fit_transform(df[['Time']].values.reshape(-1, 1))
            features_df['Time_scaled'] = time_scaled.flatten()
            feature_cols.append('Time_scaled')
        
        self.logger.info(f"Prepared {len(feature_cols)} features: {feature_cols}")
        
        return features_df, target
    
    def split_data(self, features_df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            features_df: Features DataFrame
            target: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train/test sets...")
        
        # Set random seed
        np.random.seed(self.config.seed)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            features_df,
            target,
            test_size=0.2,
            random_state=self.config.seed,
            stratify=target
        )
        
        self.logger.info(f"Train set: {X_train.shape[0]:,} samples")
        self.logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance in training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        sampling_strategy = self.config.sampling.strategy.lower()
        
        if sampling_strategy == 'none':
            self.logger.info("No sampling strategy applied")
            return X_train, y_train
        
        elif sampling_strategy == 'class_weight':
            self.logger.info("Using class_weight='balanced' strategy")
            # This will be handled in the model training phase
            return X_train, y_train
        
        elif sampling_strategy == 'under' and IMBALANCED_LEARN_AVAILABLE:
            self.logger.info("Applying RandomUnderSampler")
            self.sampler = RandomUnderSampler(
                random_state=self.config.seed,
                sampling_strategy='auto'
            )
            X_resampled, y_resampled = self.sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        
        elif sampling_strategy == 'smote' and IMBALANCED_LEARN_AVAILABLE:
            self.logger.info("Applying SMOTE")
            self.sampler = SMOTE(
                random_state=self.config.seed,
                sampling_strategy='auto'
            )
            X_resampled, y_resampled = self.sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        
        else:
            if not IMBALANCED_LEARN_AVAILABLE:
                self.logger.warning("imbalanced-learn not available, falling back to no sampling")
            else:
                self.logger.warning(f"Unknown sampling strategy: {sampling_strategy}, falling back to no sampling")
            return X_train, y_train
    
    def save_artifacts(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> None:
        """Save processed data and transformers.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        """
        self.logger.info("Saving processed data and transformers...")
        
        # Save data splits
        X_train.to_parquet(self.processed_dir / "X_train.parquet", index=False)
        X_test.to_parquet(self.processed_dir / "X_test.parquet", index=False)
        y_train.to_frame().to_parquet(self.processed_dir / "y_train.parquet", index=False)
        y_test.to_frame().to_parquet(self.processed_dir / "y_test.parquet", index=False)
        
        # Save scaler
        scaler_path = self.transformers_dir / "standard_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save sampler if used
        if self.sampler is not None:
            sampler_path = self.transformers_dir / "sampler.joblib"
            joblib.dump(self.sampler, sampler_path)
        
        self.logger.info("Artifacts saved successfully")
    
    def get_split_info(self, y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Get information about data splits.
        
        Args:
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dictionary with split information
        """
        def calculate_fraud_rate(y):
            return (y.sum() / len(y)) * 100
        
        return {
            "train": {
                "total_samples": len(y_train),
                "fraudulent_samples": int(y_train.sum()),
                "legitimate_samples": int(len(y_train) - y_train.sum()),
                "fraud_rate": calculate_fraud_rate(y_train)
            },
            "test": {
                "total_samples": len(y_test),
                "fraudulent_samples": int(y_test.sum()),
                "legitimate_samples": int(len(y_test) - y_test.sum()),
                "fraud_rate": calculate_fraud_rate(y_test)
            }
        }
    
    def preprocess_data(self) -> Dict:
        """Complete preprocessing pipeline.
        
        Returns:
            Dictionary with preprocessing results
        """
        self.logger.info("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data()
        self.logger.info(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Prepare features and target
        features_df, target = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(features_df, target)
        
        # Handle imbalance
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Save artifacts
        self.save_artifacts(X_train_resampled, X_test, y_train_resampled, y_test)
        
        # Get split information
        split_info = self.get_split_info(y_train_resampled, y_test)
        
        return {
            "split_info": split_info,
            "feature_names": list(features_df.columns),
            "sampling_strategy": self.config.sampling.strategy
        }
