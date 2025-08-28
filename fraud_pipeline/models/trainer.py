"""Model training utilities."""

import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# XGBoost availability flag
XGBOOST_AVAILABLE = False


class ModelTrainer:
    """Model trainer for fraud detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.metrics = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "fraud_detection"))
    
    def _get_model(self, model_name: str, params: Dict[str, Any]):
        """Get model instance based on name.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
        """
        if model_name == "random_forest":
            return RandomForestClassifier(**params)
        elif model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(**params)
            except ImportError:
                raise ImportError("XGBoost is not available. Please install it with: pip install xgboost")
        elif model_name == "logistic_regression":
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train a single model.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with training results
        """
        # Get model configuration
        model_config = next((m for m in self.config.get("models", []) if m["name"] == model_name), None)
        if not model_config:
            raise ValueError(f"Model configuration not found: {model_name}")
        
        # Create model
        model = self._get_model(model_name, model_config["params"])
        
        # Train model
        with mlflow.start_run(run_name=f"train_{model_name}"):
            # Log parameters
            mlflow.log_params(model_config["params"])
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            report = classification_report(y_val, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_val, y_pred)
            
            # Log metrics
            mlflow.log_metric("f1_score", report["1"]["f1-score"])
            mlflow.log_metric("precision", report["1"]["precision"])
            mlflow.log_metric("recall", report["1"]["recall"])
            mlflow.log_metric("accuracy", report["accuracy"])
            
            # Save model
            model_path = Path(self.config["data"]["models_path"]) / f"{model_name}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            
            # Log model
            mlflow.log_artifact(str(model_path))
        
        # Store results
        self.models[model_name] = model
        self.metrics[model_name] = {
            "f1_score": report["1"]["f1-score"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "accuracy": report["accuracy"],
            "confusion_matrix": conf_matrix.tolist()
        }
        
        return self.metrics[model_name]
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        for model_config in self.config.get("models", []):
            model_name = model_config["name"]
            print(f"Training {model_name}...")
            
            try:
                results[model_name] = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                print(f"✓ {model_name} trained successfully")
            except Exception as e:
                print(f"✗ Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
