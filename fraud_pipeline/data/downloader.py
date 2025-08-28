"""Data downloader for fraud detection pipeline."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ..utils.logger import setup_logger


class DataDownloader:
    """Data downloader for fraud detection dataset."""
    
    def __init__(self, config):
        """Initialize data downloader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("data_downloader")
        self.raw_dir = Path(config.data.raw_path).parent
        self.csv_path = self.raw_dir / "creditcard.csv"
        self.parquet_path = self.raw_dir / "creditcard.parquet"
    
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are available.
        
        Returns:
            True if credentials are available
        """
        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")
        
        if kaggle_username and kaggle_key:
            return True
        
        # Check if kaggle.json exists
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists()
    
    def download_from_kaggle(self) -> bool:
        """Download dataset from Kaggle API.
        
        Returns:
            True if download successful
        """
        self.logger.info("Downloading dataset from Kaggle...")
        
        try:
            # Create output directory
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the dataset
            cmd = [
                "kaggle", "datasets", "download", 
                "-d", "mlg-ulb/creditcardfraud",
                "-p", str(self.raw_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.error(f"Kaggle download failed: {result.stderr}")
                return False
            
            self.logger.info("Dataset downloaded successfully")
            
            # Extract the zip file
            zip_file = self.raw_dir / "creditcardfraud.zip"
            if zip_file.exists():
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
                
                # Remove the zip file
                zip_file.unlink()
                self.logger.info("Dataset extracted successfully")
                return True
            else:
                self.logger.error("Downloaded zip file not found")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Kaggle download timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error downloading from Kaggle: {e}")
            return False
    
    def copy_local_file(self, source_path: str) -> bool:
        """Copy local file to output directory.
        
        Args:
            source_path: Path to source file
            
        Returns:
            True if copy successful
        """
        source = Path(source_path)
        
        if not source.exists():
            self.logger.error(f"Source file not found: {source_path}")
            return False
        
        self.logger.info(f"Copying local file: {source_path}")
        
        try:
            # Create output directory
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, self.csv_path)
            self.logger.info("Local file copied successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error copying local file: {e}")
            return False
    
    def validate_dataset(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Validate the dataset structure and content.
        
        Returns:
            Tuple of (success, dataframe)
        """
        self.logger.info("Validating dataset...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_path)
            
            # Check expected columns
            expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
            missing_columns = set(expected_columns) - set(df.columns)
            
            if missing_columns:
                self.logger.error(f"Missing expected columns: {missing_columns}")
                return False, None
            
            # Check data types
            if df['Class'].dtype not in ['int64', 'int32']:
                self.logger.error("Class column should be integer type")
                return False, None
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Check class distribution
            class_counts = df['Class'].value_counts().sort_index()
            total_rows = len(df)
            fraud_rate = (class_counts.get(1, 0) / total_rows) * 100
            
            self.logger.info(f"Dataset validation successful:")
            self.logger.info(f"  - Total rows: {total_rows:,}")
            self.logger.info(f"  - Columns: {len(df.columns)}")
            self.logger.info(f"  - Class distribution: {class_counts.to_dict()}")
            self.logger.info(f"  - Fraud rate: {fraud_rate:.4f}%")
            
            return True, df
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {e}")
            return False, None
    
    def save_parquet(self, df: pd.DataFrame) -> bool:
        """Save dataset as Parquet file.
        
        Args:
            df: DataFrame to save
            
        Returns:
            True if save successful
        """
        try:
            df.to_parquet(self.parquet_path, index=False)
            self.logger.info(f"Parquet file saved: {self.parquet_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving Parquet file: {e}")
            return False
    
    def download_data(self, force: bool = False) -> bool:
        """Download and prepare the dataset.
        
        Args:
            force: Force re-download even if files exist
            
        Returns:
            True if successful
        """
        # Check if files already exist
        if not force and self.csv_path.exists() and self.parquet_path.exists():
            self.logger.info("Dataset files already exist. Use force=True to re-download.")
            
            # Validate existing files
            success, _ = self.validate_dataset()
            if success:
                self.logger.info("✅ Existing dataset is valid")
                return True
            else:
                self.logger.error("❌ Existing dataset is invalid")
                return False
        
        # Determine data source
        data_path = os.getenv("DATA_PATH")
        
        if data_path and Path(data_path).exists():
            # Copy local file
            self.logger.info("Using local DATA_PATH")
            if not self.copy_local_file(data_path):
                return False
        elif self.check_kaggle_credentials():
            # Download from Kaggle
            self.logger.info("Using Kaggle API")
            if not self.download_from_kaggle():
                return False
        else:
            self.logger.error("No data source available:")
            self.logger.error("  - DATA_PATH not set or file not found")
            self.logger.error("  - Kaggle credentials not available")
            self.logger.error("Please set DATA_PATH or configure Kaggle credentials")
            return False
        
        # Validate the dataset
        success, df = self.validate_dataset()
        if not success:
            self.logger.error("❌ Dataset validation failed")
            return False
        
        # Save Parquet version
        if not self.save_parquet(df):
            return False
        
        return True
    
    def get_dataset_info(self) -> dict:
        """Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if not self.csv_path.exists():
            return {"error": "Dataset not found"}
        
        try:
            df = pd.read_csv(self.csv_path)
            
            class_counts = df['Class'].value_counts().sort_index()
            total_rows = len(df)
            fraud_rate = (class_counts.get(1, 0) / total_rows) * 100
            
            return {
                "total_rows": total_rows,
                "columns": len(df.columns),
                "class_distribution": class_counts.to_dict(),
                "fraud_rate": fraud_rate,
                "imbalance_ratio": class_counts.get(0, 0) / class_counts.get(1, 0) if class_counts.get(1, 0) > 0 else float('inf'),
                "csv_path": str(self.csv_path),
                "parquet_path": str(self.parquet_path)
            }
        except Exception as e:
            return {"error": str(e)}
