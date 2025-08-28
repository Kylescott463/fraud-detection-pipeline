"""Configuration management for the fraud detection pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data configuration settings."""
    
    raw_path: str = Field(default="data/raw/creditcard.csv")
    processed_path: str = Field(default="data/processed/")
    models_path: str = Field(default="data/models/")
    reports_path: str = Field(default="data/reports/")


class SamplingConfig(BaseModel):
    """Sampling strategy configuration."""
    
    strategy: str = Field(default="none")
    include_time: bool = Field(default=False)
    random_state: int = Field(default=42)


class ModelParams(BaseModel):
    """Model parameters configuration."""
    
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    C: Optional[float] = None
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """Individual model configuration."""
    
    name: str
    params: ModelParams


class ServingConfig(BaseModel):
    """Model serving configuration."""
    
    threshold: float = Field(default=0.5)
    min_probability: float = Field(default=0.1)
    min_recall: float = Field(default=0.90)


class TrainingConfig(BaseModel):
    """Training configuration."""
    
    test_size: float = Field(default=0.2)
    validation_size: float = Field(default=0.2)
    cv_folds: int = Field(default=5)
    scoring: str = Field(default="f1")
    n_trials: int = Field(default=50)


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    
    tracking_uri: str = Field(default="sqlite:///mlflow.db")
    experiment_name: str = Field(default="fraud_detection")
    artifact_location: str = Field(default="data/models/")


class APIConfig(BaseModel):
    """FastAPI configuration."""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)
    workers: int = Field(default=1)


class Config(BaseModel):
    """Main configuration class."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = Field(default=42)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    models: List[ModelConfig] = Field(default_factory=list)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Config object with default settings
    """
    return Config()
