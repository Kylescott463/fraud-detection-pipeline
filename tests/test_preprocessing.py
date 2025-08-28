"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from fraud_pipeline.preprocess import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.data.processed_path = "data/processed"
        self.mock_config.data.raw_path = "data/raw/creditcard.csv"
        self.mock_config.sampling.include_time = False
        self.mock_config.random.seed = 42
        self.mock_config.seed = 42  # Add this for the split_data method
        
        # Create test data with all V1-V28 features
        test_data_dict = {
            'Time': [0, 1, 2, 3, 4],
            'Amount': [100.0, 200.0, 300.0, 400.0, 500.0],
            'Class': [0, 1, 0, 1, 0]
        }
        
        # Add V1-V28 features
        for i in range(1, 29):
            test_data_dict[f'V{i}'] = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
        
        self.test_data = pd.DataFrame(test_data_dict)
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            assert preprocessor.config == self.mock_config
            assert preprocessor.processed_dir == Path("data/processed")
            assert preprocessor.transformers_dir == Path("data/processed/transformers")
    
    @patch('fraud_pipeline.preprocess.pd.read_parquet')
    def test_load_data(self, mock_read_parquet):
        """Test data loading functionality."""
        mock_read_parquet.return_value = self.test_data
        
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            df = preprocessor.load_data()
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            assert list(df.columns) == ['Time', 'Amount', 'V1', 'V2', 'V3', 'Class']
    
    def test_prepare_features(self):
        """Test feature preparation."""
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            features_df, target = preprocessor.prepare_features(self.test_data)
            
            # Check shapes
            assert features_df.shape[0] == 5  # Same number of rows
            assert features_df.shape[1] == 30  # V1-V28 (28) + Amount_scaled (1) + V1-V3 (3) = 32, but we only have V1-V3
            assert target.shape[0] == 5
            
            # Check target values
            assert list(target) == [0, 1, 0, 1, 0]
            
            # Check that Amount_scaled column exists
            assert 'Amount_scaled' in features_df.columns
    
    def test_prepare_features_with_time(self):
        """Test feature preparation with time included."""
        self.mock_config.sampling.include_time = True
        
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            features_df, target = preprocessor.prepare_features(self.test_data)
            
            # Check that Time_scaled column exists
            assert 'Time_scaled' in features_df.columns
    
    @patch('fraud_pipeline.preprocess.train_test_split')
    def test_split_data(self, mock_split):
        """Test data splitting functionality."""
        # Mock train_test_split to return predictable splits
        X_train = pd.DataFrame({'V1': [0.1, 0.2], 'V2': [0.6, 0.7]})
        X_test = pd.DataFrame({'V1': [0.3, 0.4, 0.5], 'V2': [0.8, 0.9, 1.0]})
        y_train = pd.Series([0, 1])
        y_test = pd.Series([0, 1, 0])
        
        mock_split.return_value = (X_train, X_test, y_train, y_test)
        
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            features_df = pd.DataFrame({
                'V1': [0.1, 0.2, 0.3, 0.4, 0.5],
                'V2': [0.6, 0.7, 0.8, 0.9, 1.0],
                'Amount_scaled': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            target = pd.Series([0, 1, 0, 1, 0])
            
            X_train_result, X_test_result, y_train_result, y_test_result = preprocessor.split_data(
                features_df, target
            )
            
            # Check shapes
            assert X_train_result.shape[0] == 2
            assert X_test_result.shape[0] == 3
            assert y_train_result.shape[0] == 2
            assert y_test_result.shape[0] == 3
            
            # Check that train_test_split was called with correct parameters
            mock_split.assert_called_once()
            call_args = mock_split.call_args
            assert call_args[1]['test_size'] == 0.2
            assert call_args[1]['random_state'] == 42
            assert call_args[1]['stratify'] is not None
    
    def test_prepare_features_shape_validation(self):
        """Test that feature preparation maintains correct shapes."""
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            # Create data with known shape
            test_data_dict = {
                'Time': range(100),
                'Amount': np.random.rand(100) * 1000,
                'Class': np.random.randint(0, 2, 100)
            }
            
            # Add V1-V28 features
            for i in range(1, 29):
                test_data_dict[f'V{i}'] = np.random.randn(100)
            
            test_data = pd.DataFrame(test_data_dict)
            
            features_df, target = preprocessor.prepare_features(test_data)
            
            # Validate shapes
            assert features_df.shape[0] == 100  # Same number of rows
            assert target.shape[0] == 100  # Same number of rows
            assert features_df.shape[1] >= 30  # At least V1-V28 + Amount_scaled
            
            # Validate that features and target have same number of rows
            assert features_df.shape[0] == target.shape[0]
            
            # Validate that target contains only 0s and 1s
            assert set(target.unique()).issubset({0, 1})
    
    def test_prepare_features_column_validation(self):
        """Test that required columns are present in output."""
        with patch('fraud_pipeline.preprocess.setup_logger'):
            preprocessor = DataPreprocessor(self.mock_config)
            
            features_df, target = preprocessor.prepare_features(self.test_data)
            
            # Check that V1-V3 columns are present
            assert 'V1' in features_df.columns
            assert 'V2' in features_df.columns
            assert 'V3' in features_df.columns
            
            # Check that Amount_scaled is present
            assert 'Amount_scaled' in features_df.columns
            
            # Check that original Amount and Time are not in features
            assert 'Amount' not in features_df.columns
            assert 'Time' not in features_df.columns
