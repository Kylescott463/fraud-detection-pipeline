"""Tests for data loader module."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from fraud_pipeline.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        data_path = "test_data.csv"
        loader = DataLoader(data_path)
        
        assert loader.data_path == Path(data_path)
    
    def test_load_data_success(self):
        """Test successful data loading."""
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Amount,V1,V2,V3,Class\n")
            f.write("0,100.0,0.1,0.2,0.3,0\n")
            f.write("1,200.0,0.4,0.5,0.6,1\n")
            temp_file = f.name
        
        try:
            loader = DataLoader(temp_file)
            df = loader.load_data()
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ["Time", "Amount", "V1", "V2", "V3", "Class"]
            assert df.iloc[0]["Amount"] == 100.0
            assert df.iloc[1]["Class"] == 1
            
        finally:
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self):
        """Test data loading with non-existent file."""
        loader = DataLoader("non_existent_file.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_get_data_info(self):
        """Test getting data information."""
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Amount,V1,V2,Class\n")
            f.write("0,100.0,0.1,0.2,0\n")
            f.write("1,200.0,0.4,0.5,1\n")
            f.write("2,300.0,0.7,0.8,0\n")
            temp_file = f.name
        
        try:
            loader = DataLoader(temp_file)
            info = loader.get_data_info()
            
            assert isinstance(info, dict)
            assert info["shape"] == (3, 5)
            assert info["columns"] == ["Time", "Amount", "V1", "V2", "Class"]
            assert info["target_distribution"] == {0: 2, 1: 1}
            
        finally:
            os.unlink(temp_file)
    
    def test_get_data_info_with_missing_values(self):
        """Test getting data info with missing values."""
        # Create temporary test data with missing values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Amount,V1,V2,Class\n")
            f.write("0,100.0,0.1,,0\n")
            f.write("1,200.0,0.4,0.5,1\n")
            f.write("2,,0.7,0.8,0\n")
            temp_file = f.name
        
        try:
            loader = DataLoader(temp_file)
            info = loader.get_data_info()
            
            assert info["missing_values"]["V2"] == 1
            assert info["missing_values"]["Amount"] == 1
            assert info["missing_values"]["Time"] == 0
            
        finally:
            os.unlink(temp_file)
