"""Data loading utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd


class DataLoader:
    """Data loader for fraud detection dataset."""
    
    def __init__(self, data_path: str):
        """Initialize data loader.
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = Path(data_path)
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        return pd.read_csv(self.data_path)
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        df = self.load_data()
        
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df.get("Class", pd.Series()).value_counts().to_dict()
        }
