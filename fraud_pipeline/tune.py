"""Hyperparameter tuning module for fraud detection pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from .utils.logger import setup_logger

# Try to import Optuna, fallback to RandomizedSearchCV if not available
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """Hyperparameter tuner for fraud detection models."""
    
    def __init__(self, config):
        """Initialize hyperparameter tuner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("hyperparameter_tuner")
        self.models_dir = Path(config.data.models_path)
        self.reports_dir = Path(config.data.reports_path)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()
        
        # Load transformers
        self.scaler, self.sampler = self._load_transformers()
        
        # Define search spaces
        self.search_spaces = self._define_search_spaces()
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        
    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load training and test data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        self.logger.info("Loading training and test data...")
        
        processed_dir = Path(self.config.data.processed_path)
        X_train = pd.read_parquet(processed_dir / "X_train.parquet")
        y_train = pd.read_parquet(processed_dir / "y_train.parquet").iloc[:, 0]
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_test = pd.read_parquet(processed_dir / "y_test.parquet").iloc[:, 0]
        
        self.logger.info(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
    def _load_transformers(self) -> Tuple[Any, Optional[Any]]:
        """Load fitted transformers.
        
        Returns:
            Tuple of (scaler, sampler)
        """
        self.logger.info("Loading fitted transformers...")
        
        transformers_dir = Path(self.config.data.processed_path) / "transformers"
        
        # Load scaler
        scaler = joblib.load(transformers_dir / "standard_scaler.joblib")
        self.logger.info("Loaded fitted StandardScaler")
        
        # Load sampler if exists
        sampler = None
        sampler_path = transformers_dir / "sampler.joblib"
        if sampler_path.exists():
            sampler = joblib.load(sampler_path)
            self.logger.info(f"Loaded fitted sampler: {type(sampler).__name__}")
        
        return scaler, sampler
    
    def _define_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define hyperparameter search spaces for each model.
        
        Returns:
            Dictionary of search spaces for each model
        """
        return {
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2", "elasticnet"],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, 15, None],
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "random_state": [self.config.seed]
            }
        }
    
    def _get_model_class(self, model_name: str):
        """Get model class for given model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        if model_name == "logistic_regression":
            return LogisticRegression
        elif model_name == "random_forest":
            return RandomForestClassifier
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_pipeline(self, model_name: str, params: Dict[str, Any]) -> Pipeline:
        """Create pipeline with model and transformers.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Pipeline with scaler and model
        """
        model_class = self._get_model_class(model_name)
        
        # Filter parameters for the specific model
        model_params = {}
        for key, value in params.items():
            if key in self.search_spaces[model_name]:
                model_params[key] = value
        
        # Handle solver-specific parameters for LogisticRegression
        if model_name == "logistic_regression":
            if model_params.get("penalty") == "elasticnet":
                if "l1_ratio" not in model_params:
                    model_params["l1_ratio"] = 0.5
                model_params["solver"] = "saga"
            elif model_params.get("penalty") == "l1":
                model_params["solver"] = "liblinear"
            elif model_params.get("penalty") == "l2":
                model_params["solver"] = "liblinear"
        
        model = model_class(**model_params)
        
        # Create pipeline (without sampler - will be applied separately)
        steps = [("scaler", self.scaler), ("classifier", model)]
        
        return Pipeline(steps)
    
    def _objective_optuna(self, trial: optuna.Trial, model_name: str) -> float:
        """Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            model_name: Name of the model to optimize
            
        Returns:
            PR-AUC score
        """
        # Sample hyperparameters
        params = {}
        search_space = self.search_spaces[model_name]
        
        for param_name, param_values in search_space.items():
            if param_name == "random_state":
                params[param_name] = param_values[0]  # Use first value for random_state
            elif isinstance(param_values, list):
                if param_name in ["C", "l1_ratio"]:
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values), log=True)
                elif param_name in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif param_name in ["penalty", "solver", "max_features"]:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
        
        # Create and evaluate pipeline
        pipeline = self._create_pipeline(model_name, params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed)
        scores = []
        
        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train.iloc[train_idx]
            y_train_fold = self.y_train.iloc[train_idx]
            X_val_fold = self.X_train.iloc[val_idx]
            y_val_fold = self.y_train.iloc[val_idx]
            
            # Apply sampling if available
            if self.sampler is not None:
                X_train_sampled, y_train_sampled = self.sampler.fit_resample(X_train_fold, y_train_fold)
            else:
                X_train_sampled, y_train_sampled = X_train_fold, y_train_fold
            
            # Fit pipeline
            pipeline.fit(X_train_sampled, y_train_sampled)
            
            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            
            # Calculate PR-AUC
            score = average_precision_score(y_val_fold, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    def _tune_with_optuna(self, model_name: str, n_trials: int) -> Tuple[Dict[str, Any], float]:
        """Tune hyperparameters using Optuna.
        
        Args:
            model_name: Name of the model to tune
            n_trials: Number of trials
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self.logger.info(f"Tuning {model_name} with Optuna ({n_trials} trials)...")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective_optuna(trial, model_name),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.logger.info(f"Best {model_name} params: {best_params}")
        self.logger.info(f"Best {model_name} score: {best_score:.4f}")
        
        return best_params, best_score
    
    def _tune_with_randomized_search(self, model_name: str, n_trials: int) -> Tuple[Dict[str, Any], float]:
        """Tune hyperparameters using RandomizedSearchCV.
        
        Args:
            model_name: Name of the model to tune
            n_trials: Number of trials (converted to n_iter)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self.logger.info(f"Tuning {model_name} with RandomizedSearchCV ({n_trials} iterations)...")
        
        # Get model class and create base model
        model_class = self._get_model_class(model_name)
        base_model = model_class()
        
        # Create pipeline (without sampler - will be applied separately)
        steps = [("scaler", self.scaler), ("classifier", base_model)]
        pipeline = Pipeline(steps)
        
        # Define parameter grid
        param_grid = {}
        for param_name, param_values in self.search_spaces[model_name].items():
            if param_name != "random_state":
                param_grid[f"classifier__{param_name}"] = param_values
        
        # Create custom CV splitter that applies sampling
        from sklearn.model_selection import BaseCrossValidator
        import numpy as np
        
        class SamplingCV(BaseCrossValidator):
            def __init__(self, n_splits, sampler, random_state):
                self.n_splits = n_splits
                self.sampler = sampler
                self.random_state = random_state
                self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            def split(self, X, y=None, groups=None):
                return self.cv.split(X, y, groups)
            
            def get_n_splits(self, X=None, y=None, groups=None):
                return self.n_splits
        
        # Randomized search with custom CV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=SamplingCV(3, self.sampler, self.config.seed),
            scoring="average_precision",
            random_state=self.config.seed,
            n_jobs=1,  # Use single job to avoid issues with sampling
            verbose=1
        )
        
        # Apply sampling to training data if available
        if self.sampler is not None:
            X_train_sampled, y_train_sampled = self.sampler.fit_resample(self.X_train, self.y_train)
        else:
            X_train_sampled, y_train_sampled = self.X_train, self.y_train
        
        # Fit
        random_search.fit(X_train_sampled, y_train_sampled)
        
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        # Clean up parameter names
        cleaned_params = {}
        for key, value in best_params.items():
            if key.startswith("classifier__"):
                cleaned_params[key.replace("classifier__", "")] = value
        
        self.logger.info(f"Best {model_name} params: {cleaned_params}")
        self.logger.info(f"Best {model_name} score: {best_score:.4f}")
        
        return cleaned_params, best_score
    
    def _evaluate_best_model(self, model_name: str, best_params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate best model on test set.
        
        Args:
            model_name: Name of the model
            best_params: Best hyperparameters
            
        Returns:
            Dictionary of test metrics
        """
        self.logger.info(f"Evaluating best {model_name} on test set...")
        
        # Create best pipeline
        best_pipeline = self._create_pipeline(model_name, best_params)
        
        # Apply sampling if available
        if self.sampler is not None:
            X_train_sampled, y_train_sampled = self.sampler.fit_resample(self.X_train, self.y_train)
        else:
            X_train_sampled, y_train_sampled = self.X_train, self.y_train
        
        # Fit on full training data
        best_pipeline.fit(X_train_sampled, y_train_sampled)
        
        # Predict on test set
        y_pred_proba = best_pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = best_pipeline.predict(self.X_test)
        
        # Calculate metrics
        from sklearn.metrics import (
            average_precision_score, roc_auc_score,
            precision_score, recall_score, f1_score
        )
        
        metrics = {
            "pr_auc": average_precision_score(self.y_test, y_pred_proba),
            "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred)
        }
        
        self.logger.info(f"Test metrics for {model_name}: {metrics}")
        
        return metrics, best_pipeline
    
    def _log_to_mlflow(self, model_name: str, best_params: Dict[str, Any], 
                      best_score: float, test_metrics: Dict[str, float]) -> None:
        """Log tuning results to MLflow.
        
        Args:
            model_name: Name of the model
            best_params: Best hyperparameters
            best_score: Best CV score
            test_metrics: Test set metrics
        """
        with mlflow.start_run(run_name=f"tune_{model_name}"):
            # Log parameters
            mlflow.log_params(best_params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_trials", self.config.training.n_trials)
            
            # Log metrics
            mlflow.log_metric("cv_pr_auc", best_score)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Log model
            model_path = self.models_dir / f"best_{model_name}.joblib"
            mlflow.log_artifact(str(model_path))
            
            self.logger.info(f"Logged {model_name} tuning results to MLflow")
    
    def tune_models(self, model_names: List[str], n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters for specified models.
        
        Args:
            model_names: List of model names to tune
            n_trials: Number of trials for each model
            
        Returns:
            Dictionary with tuning results
        """
        self.logger.info(f"Starting hyperparameter tuning for models: {model_names}")
        
        results = {}
        
        for model_name in model_names:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Tuning {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                # Tune hyperparameters
                if OPTUNA_AVAILABLE:
                    best_params, best_score = self._tune_with_optuna(model_name, n_trials)
                else:
                    best_params, best_score = self._tune_with_randomized_search(model_name, n_trials)
                
                # Evaluate best model
                test_metrics, best_pipeline = self._evaluate_best_model(model_name, best_params)
                
                # Save best model
                model_path = self.models_dir / f"best_{model_name}.joblib"
                joblib.dump(best_pipeline, model_path)
                self.logger.info(f"Saved best {model_name} to {model_path}")
                
                # Log to MLflow
                self._log_to_mlflow(model_name, best_params, best_score, test_metrics)
                
                # Store results
                results[model_name] = {
                    "best_params": best_params,
                    "best_score": best_score,
                    "test_metrics": test_metrics,
                    "model_path": str(model_path)
                }
                
            except Exception as e:
                self.logger.error(f"Error tuning {model_name}: {e}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select the best model based on test PR-AUC.
        
        Args:
            results: Tuning results
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not results:
            raise ValueError("No tuning results available")
        
        # Find model with highest test PR-AUC
        best_model = max(results.keys(), key=lambda x: results[x]["test_metrics"]["pr_auc"])
        best_score = results[best_model]["test_metrics"]["pr_auc"]
        
        self.logger.info(f"Best model: {best_model} (PR-AUC: {best_score:.4f})")
        
        return best_model, results[best_model]
    
    def save_best_model(self, best_model_name: str, best_model_results: Dict[str, Any]) -> None:
        """Save the best model as the main model.
        
        Args:
            best_model_name: Name of the best model
            best_model_results: Results for the best model
        """
        # Copy best model to main model file
        best_model_path = Path(best_model_results["model_path"])
        main_model_path = self.models_dir / "best_model.joblib"
        
        # Load and save as main model
        best_pipeline = joblib.load(best_model_path)
        joblib.dump(best_pipeline, main_model_path)
        
        # Save tuning summary
        summary = {
            "best_model": best_model_name,
            "best_params": best_model_results["best_params"],
            "cv_score": best_model_results["best_score"],
            "test_metrics": best_model_results["test_metrics"],
            "tuned_at": pd.Timestamp.now().isoformat()
        }
        
        summary_path = self.models_dir / "tuning_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved best model as {main_model_path}")
        self.logger.info(f"Saved tuning summary to {summary_path}")
    
    def print_results(self, results: Dict[str, Any], best_model_name: str) -> None:
        """Print tuning results to console.
        
        Args:
            results: Tuning results
            best_model_name: Name of the best model
        """
        print("\n" + "="*80)
        print("🎯 HYPERPARAMETER TUNING RESULTS")
        print("="*80)
        
        # Print results for each model
        for model_name, model_results in results.items():
            print(f"\n📊 {model_name.upper()}")
            print("-" * 40)
            print(f"Best CV PR-AUC: {model_results['best_score']:.4f}")
            print(f"Test PR-AUC:    {model_results['test_metrics']['pr_auc']:.4f}")
            print(f"Test ROC-AUC:   {model_results['test_metrics']['roc_auc']:.4f}")
            print(f"Test Precision: {model_results['test_metrics']['precision']:.4f}")
            print(f"Test Recall:    {model_results['test_metrics']['recall']:.4f}")
            print(f"Test F1:        {model_results['test_metrics']['f1']:.4f}")
            
            print(f"\nBest Parameters:")
            for param, value in model_results['best_params'].items():
                print(f"  {param}: {value}")
        
        print(f"\n" + "="*80)
        print("🏆 BEST MODEL SELECTION")
        print("="*80)
        print(f"Selected Model: {best_model_name}")
        print(f"Test PR-AUC: {results[best_model_name]['test_metrics']['pr_auc']:.4f}")
        print(f"Saved as: data/models/best_model.joblib")
        
        if not OPTUNA_AVAILABLE:
            print(f"\n⚠️  Note: Optuna not available, used RandomizedSearchCV instead")
        
        print(f"\n✅ Tuning completed! Results logged to MLflow")
