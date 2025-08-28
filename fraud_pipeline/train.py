"""Model training module for fraud detection pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost availability flag
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False

from .utils.logger import setup_logger
from .utils.metrics import calculate_metrics, format_metrics


class ModelTrainer:
    """Model trainer for fraud detection pipeline."""
    
    def __init__(self, config):
        """Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("model_trainer")
        self.models_dir = Path(config.data.models_path)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        
        # Load data
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()
        
        # Load transformers
        self.scaler, self.sampler = self._load_transformers()
        
        # Initialize models
        self.models = self._initialize_models()
        
    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load training and test data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        self.logger.info("Loading training and test data...")
        
        processed_dir = Path(self.config.data.processed_path)
        
        X_train = pd.read_parquet(processed_dir / "X_train.parquet")
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_train = pd.read_parquet(processed_dir / "y_train.parquet").iloc[:, 0]
        y_test = pd.read_parquet(processed_dir / "y_test.parquet").iloc[:, 0]
        
        self.logger.info(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def _load_transformers(self) -> Tuple[Optional[StandardScaler], Optional[Any]]:
        """Load fitted transformers.
        
        Returns:
            Tuple of (scaler, sampler)
        """
        transformers_dir = Path(self.config.data.processed_path) / "transformers"
        
        # Load scaler
        scaler = None
        scaler_path = transformers_dir / "standard_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            self.logger.info("Loaded fitted StandardScaler")
        
        # Load sampler
        sampler = None
        sampler_path = transformers_dir / "sampler.joblib"
        if sampler_path.exists():
            sampler = joblib.load(sampler_path)
            self.logger.info(f"Loaded fitted sampler: {type(sampler).__name__}")
        
        return scaler, sampler
    
    def _initialize_models(self) -> Dict[str, Dict]:
        """Initialize model configurations.
        
        Returns:
            Dictionary of model configurations
        """
        models = {}
        
        # Logistic Regression
        models["logistic_regression"] = {
            "model": LogisticRegression(
                solver="liblinear",
                random_state=self.config.seed,
                max_iter=1000
            ),
            "params": {
                "solver": "liblinear",
                "random_state": self.config.seed,
                "max_iter": 1000
            }
        }
        
        # Random Forest
        models["random_forest"] = {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.seed,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": self.config.seed
            }
        }
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models["xgboost"] = {
                "model": XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.seed,
                    n_jobs=-1
                ),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": self.config.seed
                }
            }
        else:
            self.logger.warning("XGBoost not available. Skipping XGBoost model.")
        
        return models
    
    def _create_pipeline(self, model_name: str, model: Any) -> Pipeline:
        """Create sklearn pipeline for model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            
        Returns:
            sklearn Pipeline
        """
        steps = []
        
        # Add scaler if needed (for linear models)
        if model_name == "logistic_regression":
            if self.scaler is not None:
                steps.append(("scaler", self.scaler))
        
        # Add classifier
        steps.append(("classifier", model))
        
        return Pipeline(steps)
    
    def _apply_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply sampling to training data if needed.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        if self.sampler is not None and self.config.sampling.strategy != "class_weight":
            self.logger.info(f"Applying {type(self.sampler).__name__} to training data")
            X_resampled, y_resampled = self.sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        else:
            return X_train, y_train
    
    def _calculate_precision_recall_at_k(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: float = 0.1) -> Tuple[float, float]:
        """Calculate precision and recall at k% threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            k: Percentage threshold (default 0.1 = 10%)
            
        Returns:
            Tuple of (precision_at_k, recall_at_k)
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_true = y_true[sorted_indices]
        
        # Calculate k% threshold
        k_threshold = int(len(y_true) * k)
        
        # Calculate precision and recall at k%
        if k_threshold > 0:
            precision_at_k = np.sum(sorted_true[:k_threshold]) / k_threshold
            recall_at_k = np.sum(sorted_true[:k_threshold]) / np.sum(y_true)
        else:
            precision_at_k = 0.0
            recall_at_k = 0.0
        
        return precision_at_k, recall_at_k
    
    def _evaluate_model(self, model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model with cross-validation and test set.
        
        Args:
            model_name: Name of the model
            pipeline: Fitted pipeline
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        
        # CV scores for different metrics
        cv_pr_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision')
        cv_roc_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        
        # Test set predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_pr_auc = average_precision_score(y_test, y_pred_proba)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        
        # Precision/Recall at k%
        precision_at_10, recall_at_10 = self._calculate_precision_recall_at_k(y_test.values, y_pred_proba, k=0.1)
        precision_at_5, recall_at_5 = self._calculate_precision_recall_at_k(y_test.values, y_pred_proba, k=0.05)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "model_name": model_name,
            "cv_pr_auc_mean": cv_pr_auc.mean(),
            "cv_pr_auc_std": cv_pr_auc.std(),
            "cv_roc_auc_mean": cv_roc_auc.mean(),
            "cv_roc_auc_std": cv_roc_auc.std(),
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_precision_at_10": precision_at_10,
            "test_recall_at_10": recall_at_10,
            "test_precision_at_5": precision_at_5,
            "test_recall_at_5": recall_at_5,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
    
    def _log_to_mlflow(self, model_name: str, pipeline: Pipeline, results: Dict[str, Any], 
                      model_params: Dict[str, Any]) -> None:
        """Log model and results to MLflow.
        
        Args:
            model_name: Name of the model
            pipeline: Fitted pipeline
            results: Evaluation results
            model_params: Model parameters
        """
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("sampling_strategy", self.config.sampling.strategy)
            mlflow.log_param("include_time", self.config.sampling.include_time)
            
            # Log metrics
            mlflow.log_metric("cv_pr_auc_mean", results["cv_pr_auc_mean"])
            mlflow.log_metric("cv_pr_auc_std", results["cv_pr_auc_std"])
            mlflow.log_metric("cv_roc_auc_mean", results["cv_roc_auc_mean"])
            mlflow.log_metric("cv_roc_auc_std", results["cv_roc_auc_std"])
            mlflow.log_metric("cv_f1_mean", results["cv_f1_mean"])
            mlflow.log_metric("cv_f1_std", results["cv_f1_std"])
            mlflow.log_metric("test_pr_auc", results["test_pr_auc"])
            mlflow.log_metric("test_roc_auc", results["test_roc_auc"])
            mlflow.log_metric("test_precision", results["test_precision"])
            mlflow.log_metric("test_recall", results["test_recall"])
            mlflow.log_metric("test_f1", results["test_f1"])
            mlflow.log_metric("test_precision_at_10", results["test_precision_at_10"])
            mlflow.log_metric("test_recall_at_10", results["test_recall_at_10"])
            mlflow.log_metric("test_precision_at_5", results["test_precision_at_5"])
            mlflow.log_metric("test_recall_at_5", results["test_recall_at_5"])
            
            # Log model
            mlflow.sklearn.log_model(pipeline, f"{model_name}_pipeline")
            
            # Log confusion matrix as artifact
            cm_df = pd.DataFrame(
                results["confusion_matrix"],
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            cm_path = self.models_dir / f"{model_name}_confusion_matrix.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(str(cm_path))
            
            # Log classification report
            report = classification_report(self.y_test, results["y_pred"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = self.models_dir / f"{model_name}_classification_report.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(str(report_path))
    
    def _save_model(self, model_name: str, pipeline: Pipeline) -> None:
        """Save trained pipeline to disk.
        
        Args:
            model_name: Name of the model
            pipeline: Fitted pipeline
        """
        model_path = self.models_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        self.logger.info(f"Saved {model_name} to {model_path}")
    
    def train_models(self) -> Dict[str, Dict[str, Any]]:
        """Train all models and evaluate them.
        
        Returns:
            Dictionary with training results for each model
        """
        self.logger.info("Starting model training...")
        
        results = {}
        
        for model_name, model_config in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            # Apply sampling to training data
            X_train_sampled, y_train_sampled = self._apply_sampling(self.X_train, self.y_train)
            
            # Create pipeline
            pipeline = self._create_pipeline(model_name, model_config["model"])
            
            # Fit pipeline
            pipeline.fit(X_train_sampled, y_train_sampled)
            
            # Evaluate model
            model_results = self._evaluate_model(
                model_name, pipeline, X_train_sampled, y_train_sampled, 
                self.X_test, self.y_test
            )
            
            # Log to MLflow
            self._log_to_mlflow(model_name, pipeline, model_results, model_config["params"])
            
            # Save model
            self._save_model(model_name, pipeline)
            
            results[model_name] = model_results
        
        return results
    
    def print_results_table(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print results table to console.
        
        Args:
            results: Training results
        """
        print("\n" + "="*100)
        print("📊 MODEL TRAINING RESULTS")
        print("="*100)
        
        # Header
        print(f"{'Model':<20} {'CV PR-AUC':<12} {'CV ROC-AUC':<12} {'CV F1':<10} {'Test PR-AUC':<12} {'Test ROC-AUC':<12} {'Test F1':<10}")
        print("-" * 100)
        
        # Results
        for model_name, result in results.items():
            print(f"{model_name:<20} "
                  f"{result['cv_pr_auc_mean']:.4f}±{result['cv_pr_auc_std']:.3f} "
                  f"{result['cv_roc_auc_mean']:.4f}±{result['cv_roc_auc_std']:.3f} "
                  f"{result['cv_f1_mean']:.4f}±{result['cv_f1_std']:.3f} "
                  f"{result['test_pr_auc']:.4f} "
                  f"{result['test_roc_auc']:.4f} "
                  f"{result['test_f1']:.4f}")
        
        print("\n" + "="*100)
        print("🎯 DETAILED TEST METRICS")
        print("="*100)
        
        for model_name, result in results.items():
            print(f"\n📈 {model_name.upper()}:")
            print(f"   PR-AUC: {result['test_pr_auc']:.4f}")
            print(f"   ROC-AUC: {result['test_roc_auc']:.4f}")
            print(f"   Precision: {result['test_precision']:.4f}")
            print(f"   Recall: {result['test_recall']:.4f}")
            print(f"   F1-Score: {result['test_f1']:.4f}")
            print(f"   Precision@10%: {result['test_precision_at_10']:.4f}")
            print(f"   Recall@10%: {result['test_recall_at_10']:.4f}")
            print(f"   Precision@5%: {result['test_precision_at_5']:.4f}")
            print(f"   Recall@5%: {result['test_recall_at_5']:.4f}")
            
            # Confusion matrix
            cm = result['confusion_matrix']
            print(f"   Confusion Matrix:")
            print(f"   [[{cm[0,0]:>6} {cm[0,1]:>6}]")
            print(f"    [{cm[1,0]:>6} {cm[1,1]:>6}]]")
        
        print(f"\n✅ Training completed! Models saved to: {self.models_dir}")
        print(f"📊 MLflow tracking URI: {self.config.mlflow.tracking_uri}")
        print(f"🔬 Experiment: {self.config.mlflow.experiment_name}")
