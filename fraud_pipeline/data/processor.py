"""Data processing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Data processor for fraud detection pipeline."""
    
    def __init__(self, random_state: int = 42):
        """Initialize data processor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def split_data(self, df: pd.DataFrame, target_col: str = "Class", 
                   test_size: float = 0.2, val_size: float = 0.2) -> tuple:
        """Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """Fit the scaler on training data.
        
        Args:
            X_train: Training features
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
    
    def transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
            
        Raises:
            ValueError: If scaler is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def process_data(self, df: pd.DataFrame, target_col: str = "Class",
                    test_size: float = 0.2, val_size: float = 0.2) -> tuple:
        """Complete data processing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of processed data splits
        """
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df, target_col, test_size, val_size
        )
        
        # Fit scaler on training data
        self.fit_scaler(X_train)
        
        # Transform all splits
        X_train_scaled = self.transform_data(X_train)
        X_val_scaled = self.transform_data(X_val)
        X_test_scaled = self.transform_data(X_test)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test)
