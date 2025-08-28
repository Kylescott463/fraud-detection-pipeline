"""Model prediction utilities."""

import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ModelPredictor:
    """Model predictor for fraud detection."""
    
    def __init__(self, models_path: str, threshold: float = 0.5):
        """Initialize model predictor.
        
        Args:
            models_path: Path to trained models
            threshold: Prediction threshold for fraud detection
        """
        self.models_path = Path(models_path)
        self.threshold = threshold
        self.models = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all available trained models."""
        if not self.models_path.exists():
            print(f"Models directory not found: {self.models_path}")
            return
        
        for model_file in self.models_path.glob("*.joblib"):
            model_name = model_file.stem
            try:
                self.models[model_name] = joblib.load(model_file)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
    
    def predict(self, features: Union[List[float], np.ndarray, pd.DataFrame], 
               model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction using specified or best model.
        
        Args:
            features: Input features
            model_name: Name of model to use (if None, uses best model)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            raise ValueError("No models available for prediction")
        
        # Convert features to numpy array
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
        
        # Use specified model or first available
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
            model = self.models[model_name]
        else:
            model = list(self.models.values())[0]
            model_name = list(self.models.keys())[0]
        
        # Make prediction
        try:
            probability = model.predict_proba(features)[0, 1]
            prediction = 1 if probability >= self.threshold else 0
            
            return {
                "model_name": model_name,
                "prediction": int(prediction),
                "probability": float(probability),
                "is_fraud": bool(prediction),
                "threshold": self.threshold
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "prediction": None,
                "probability": None,
                "is_fraud": None
            }
    
    def predict_batch(self, features: Union[List[List[float]], np.ndarray, pd.DataFrame],
                     model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Make batch predictions.
        
        Args:
            features: Batch of input features
            model_name: Name of model to use
            
        Returns:
            List of prediction results
        """
        if isinstance(features, list):
            features = np.array(features)
        elif isinstance(features, pd.DataFrame):
            features = features.values
        
        results = []
        for i in range(len(features)):
            result = self.predict(features[i], model_name)
            result["index"] = i
            results.append(result)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        model = self.models[model_name]
        
        return {
            "name": model_name,
            "type": type(model).__name__,
            "parameters": getattr(model, "get_params", lambda: {})()
        }
