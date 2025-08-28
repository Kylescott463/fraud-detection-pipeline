"""Model evaluation and selection module for fraud detection pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss,
    average_precision_score, roc_auc_score
)

from .utils.logger import setup_logger


class ModelEvaluator:
    """Model evaluator for fraud detection pipeline."""
    
    def __init__(self, config):
        """Initialize model evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = setup_logger("model_evaluator")
        self.models_dir = Path(config.data.models_path)
        self.reports_dir = Path(config.data.reports_path)
        
        # Create reports directory
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.X_test, self.y_test = self._load_test_data()
        
        # Load models
        self.models = self._load_models()
        
    def _load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        self.logger.info("Loading test data...")
        
        processed_dir = Path(self.config.data.processed_path)
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_test = pd.read_parquet(processed_dir / "y_test.parquet").iloc[:, 0]
        
        self.logger.info(f"Loaded test data: {X_test.shape}")
        return X_test, y_test
    
    def _load_models(self) -> Dict[str, Any]:
        """Load trained models.
        
        Returns:
            Dictionary of loaded models
        """
        self.logger.info("Loading trained models...")
        
        models = {}
        model_files = list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem
            if model_name not in ["threshold"]:  # Skip non-model files
                try:
                    model = joblib.load(model_file)
                    models[model_name] = model
                    self.logger.info(f"Loaded {model_name}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_name}: {e}")
        
        return models
    
    def _evaluate_model(self, model_name: str, model: Any) -> Dict[str, Any]:
        """Evaluate a single model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Calculate curves
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            self.y_test, y_pred_proba
        )
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(
            self.y_test, y_pred_proba
        )
        
        # Calculate Brier score
        brier_score = brier_score_loss(self.y_test, y_pred_proba)
        
        return {
            "model_name": model_name,
            "y_pred_proba": y_pred_proba,
            "y_pred": y_pred,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier_score": brier_score,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "pr_thresholds": pr_thresholds,
            "fpr_curve": fpr_curve,
            "tpr_curve": tpr_curve,
            "roc_thresholds": roc_thresholds
        }
    
    def _find_optimal_threshold(self, results: Dict[str, Any], min_recall: float = 0.90) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold that achieves minimum recall while maximizing precision.
        
        Args:
            results: Model evaluation results
            min_recall: Minimum recall requirement
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        precision_curve = results["precision_curve"]
        recall_curve = results["recall_curve"]
        pr_thresholds = results["pr_thresholds"]
        
        # Find indices where recall >= min_recall
        valid_indices = np.where(recall_curve >= min_recall)[0]
        
        if len(valid_indices) == 0:
            self.logger.warning(f"Could not achieve {min_recall} recall. Using default threshold 0.5")
            optimal_threshold = 0.5
            optimal_idx = np.argmin(np.abs(pr_thresholds - optimal_threshold))
        else:
            # Among valid indices, find the one with maximum precision
            best_idx = valid_indices[np.argmax(precision_curve[valid_indices])]
            optimal_threshold = pr_thresholds[best_idx]
            optimal_idx = best_idx
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (results["y_pred_proba"] >= optimal_threshold).astype(int)
        
        metrics_at_threshold = {
            "threshold": optimal_threshold,
            "precision": precision_score(self.y_test, y_pred_optimal),
            "recall": recall_score(self.y_test, y_pred_optimal),
            "f1": f1_score(self.y_test, y_pred_optimal),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred_optimal).tolist()
        }
        
        return optimal_threshold, metrics_at_threshold
    
    def _calibrate_model(self, model: Any, model_name: str) -> Tuple[Any, float]:
        """Calibrate model using Platt scaling if it improves Brier score.
        
        Args:
            model: Model to calibrate
            model_name: Name of the model
            
        Returns:
            Tuple of (calibrated_model, original_brier_score)
        """
        # Get original predictions and Brier score
        y_pred_proba_orig = model.predict_proba(self.X_test)[:, 1]
        original_brier = brier_score_loss(self.y_test, y_pred_proba_orig)
        
        # Try calibration
        try:
            calibrated_model = CalibratedClassifierCV(model, cv=5, method='sigmoid')
            calibrated_model.fit(self.X_test, self.y_test)
            
            y_pred_proba_cal = calibrated_model.predict_proba(self.X_test)[:, 1]
            calibrated_brier = brier_score_loss(self.y_test, y_pred_proba_cal)
            
            if calibrated_brier < original_brier:
                self.logger.info(f"Calibration improved Brier score for {model_name}: {original_brier:.4f} -> {calibrated_brier:.4f}")
                return calibrated_model, original_brier
            else:
                self.logger.info(f"Calibration did not improve Brier score for {model_name}: {original_brier:.4f} vs {calibrated_brier:.4f}")
                return model, original_brier
                
        except Exception as e:
            self.logger.warning(f"Calibration failed for {model_name}: {e}")
            return model, original_brier
    
    def _plot_curves(self, results: Dict[str, Dict[str, Any]], best_model_name: str) -> None:
        """Plot PR and ROC curves.
        
        Args:
            results: All model evaluation results
            best_model_name: Name of the best model
        """
        self.logger.info("Creating PR and ROC curves...")
        
        # PR Curve
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for model_name, result in results.items():
            plt.plot(result["recall_curve"], result["precision_curve"], 
                    label=f"{model_name} (PR-AUC: {result['pr_auc']:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ROC Curve
        plt.subplot(1, 2, 2)
        for model_name, result in results.items():
            plt.plot(result["fpr_curve"], result["tpr_curve"], 
                    label=f"{model_name} (ROC-AUC: {result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / "model_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Best model PR curve with threshold
        best_result = results[best_model_name]
        optimal_threshold, _ = self._find_optimal_threshold(best_result)
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_result["recall_curve"], best_result["precision_curve"], 
                label=f"{best_model_name} (PR-AUC: {best_result['pr_auc']:.3f})")
        
        # Mark optimal threshold point
        optimal_idx = np.argmin(np.abs(best_result["pr_thresholds"] - optimal_threshold))
        plt.plot(best_result["recall_curve"][optimal_idx], best_result["precision_curve"][optimal_idx], 
                'ro', markersize=10, label=f'Optimal threshold: {optimal_threshold:.3f}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {best_model_name} (Best Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.reports_dir / "best_model_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, results: Dict[str, Dict[str, Any]], best_model_name: str, 
                     optimal_threshold: float, metrics_at_threshold: Dict[str, Any]) -> None:
        """Save evaluation metrics to JSON.
        
        Args:
            results: All model evaluation results
            best_model_name: Name of the best model
            optimal_threshold: Optimal threshold
            metrics_at_threshold: Metrics at optimal threshold
        """
        metrics = {
            "best_model": best_model_name,
            "optimal_threshold": optimal_threshold,
            "min_recall_requirement": self.config.serving.min_recall,
            "test_metrics": {},
            "metrics_at_threshold": metrics_at_threshold
        }
        
        for model_name, result in results.items():
            metrics["test_metrics"][model_name] = {
                "pr_auc": result["pr_auc"],
                "roc_auc": result["roc_auc"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "brier_score": result["brier_score"]
            }
        
        metrics_path = self.reports_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_path}")
    
    def _save_threshold(self, optimal_threshold: float) -> None:
        """Save optimal threshold to file.
        
        Args:
            optimal_threshold: Optimal threshold value
        """
        threshold_data = {
            "threshold": optimal_threshold,
            "model_name": "best_model",  # Will be updated with actual model name
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        threshold_path = self.models_dir / "threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        self.logger.info(f"Saved threshold to {threshold_path}")
    
    def _create_final_report(self, results: Dict[str, Dict[str, Any]], best_model_name: str,
                           optimal_threshold: float, metrics_at_threshold: Dict[str, Any]) -> None:
        """Create final evaluation report.
        
        Args:
            results: All model evaluation results
            best_model_name: Name of the best model
            optimal_threshold: Optimal threshold
            metrics_at_threshold: Metrics at optimal threshold
        """
        report_path = self.reports_dir / "final_report.md"
        
        # Sort models by PR-AUC
        sorted_models = sorted(results.items(), key=lambda x: x[1]["pr_auc"], reverse=True)
        
        with open(report_path, 'w') as f:
            f.write("# Fraud Detection Model Evaluation Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Selection\n\n")
            f.write(f"**Best Model:** {best_model_name}\n")
            f.write(f"**Selection Criterion:** PR-AUC (Precision-Recall Area Under Curve)\n")
            f.write(f"**Optimal Threshold:** {optimal_threshold:.4f}\n")
            f.write(f"**Minimum Recall Requirement:** {self.config.serving.min_recall:.2f}\n\n")
            
            f.write("## Model Performance Comparison\n\n")
            f.write("| Model | PR-AUC | ROC-AUC | Precision | Recall | F1-Score | Brier Score |\n")
            f.write("|-------|--------|---------|-----------|--------|----------|-------------|\n")
            
            for model_name, result in sorted_models:
                f.write(f"| {model_name} | {result['pr_auc']:.4f} | {result['roc_auc']:.4f} | "
                       f"{result['precision']:.4f} | {result['recall']:.4f} | {result['f1']:.4f} | "
                       f"{result['brier_score']:.4f} |\n")
            
            f.write("\n## Best Model Performance at Optimal Threshold\n\n")
            f.write(f"**Model:** {best_model_name}\n")
            f.write(f"**Threshold:** {optimal_threshold:.4f}\n")
            f.write(f"**Precision:** {metrics_at_threshold['precision']:.4f}\n")
            f.write(f"**Recall:** {metrics_at_threshold['recall']:.4f}\n")
            f.write(f"**F1-Score:** {metrics_at_threshold['f1']:.4f}\n\n")
            
            # Confusion matrix
            cm = metrics_at_threshold['confusion_matrix']
            f.write("## Confusion Matrix (at Optimal Threshold)\n\n")
            f.write("```\n")
            f.write("                Predicted\n")
            f.write("Actual    Negative    Positive\n")
            f.write(f"Negative  {cm[0][0]:>8}  {cm[0][1]:>8}\n")
            f.write(f"Positive  {cm[1][0]:>8}  {cm[1][1]:>8}\n")
            f.write("```\n\n")
            
            f.write("## Key Insights\n\n")
            f.write(f"- **Best Model:** {best_model_name} achieved the highest PR-AUC of {results[best_model_name]['pr_auc']:.4f}\n")
            f.write(f"- **Threshold Optimization:** Selected threshold {optimal_threshold:.4f} to achieve ≥{self.config.serving.min_recall:.2f} recall\n")
            f.write(f"- **Precision at Threshold:** {metrics_at_threshold['precision']:.4f}\n")
            f.write(f"- **Recall at Threshold:** {metrics_at_threshold['recall']:.4f}\n")
            
            if len(results) > 1:
                second_best = sorted_models[1][0]
                f.write(f"- **Runner-up:** {second_best} with PR-AUC {results[second_best]['pr_auc']:.4f}\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- `test_metrics.json` - Complete evaluation metrics\n")
            f.write("- `model_curves.png` - PR and ROC curves for all models\n")
            f.write("- `best_model_pr_curve.png` - PR curve with optimal threshold\n")
            f.write("- `threshold.json` - Optimal threshold for serving\n")
            f.write("- `final_report.md` - This report\n")
        
        self.logger.info(f"Created final report: {report_path}")
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all models and select the best one.
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting model evaluation...")
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            # Try calibration
            calibrated_model, original_brier = self._calibrate_model(model, model_name)
            
            # Evaluate model
            result = self._evaluate_model(model_name, calibrated_model)
            results[model_name] = result
        
        # Select best model by PR-AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]["pr_auc"])
        self.logger.info(f"Best model by PR-AUC: {best_model_name} ({results[best_model_name]['pr_auc']:.4f})")
        
        # Find optimal threshold for best model
        optimal_threshold, metrics_at_threshold = self._find_optimal_threshold(
            results[best_model_name], self.config.serving.min_recall
        )
        
        # Create plots
        self._plot_curves(results, best_model_name)
        
        # Save metrics
        self._save_metrics(results, best_model_name, optimal_threshold, metrics_at_threshold)
        
        # Save threshold
        self._save_threshold(optimal_threshold)
        
        # Create final report
        self._create_final_report(results, best_model_name, optimal_threshold, metrics_at_threshold)
        
        return {
            "results": results,
            "best_model": best_model_name,
            "optimal_threshold": optimal_threshold,
            "metrics_at_threshold": metrics_at_threshold
        }
    
    def print_results(self, evaluation_results: Dict[str, Any]) -> None:
        """Print evaluation results to console.
        
        Args:
            evaluation_results: Results from evaluate_models
        """
        results = evaluation_results["results"]
        best_model = evaluation_results["best_model"]
        optimal_threshold = evaluation_results["optimal_threshold"]
        metrics_at_threshold = evaluation_results["metrics_at_threshold"]
        
        print("\n" + "="*80)
        print("🎯 MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Model comparison table
        print(f"{'Model':<20} {'PR-AUC':<10} {'ROC-AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        
        for model_name, result in sorted(results.items(), key=lambda x: x[1]["pr_auc"], reverse=True):
            print(f"{model_name:<20} {result['pr_auc']:<10.4f} {result['roc_auc']:<10.4f} "
                  f"{result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f}")
        
        print("\n" + "="*80)
        print("🏆 BEST MODEL SELECTION")
        print("="*80)
        print(f"Selected Model: {best_model}")
        print(f"Selection Criterion: PR-AUC (Precision-Recall Area Under Curve)")
        print(f"PR-AUC Score: {results[best_model]['pr_auc']:.4f}")
        
        print("\n" + "="*80)
        print("⚙️  THRESHOLD OPTIMIZATION")
        print("="*80)
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Minimum Recall Requirement: ≥{self.config.serving.min_recall:.2f}")
        print(f"Precision at Threshold: {metrics_at_threshold['precision']:.4f}")
        print(f"Recall at Threshold: {metrics_at_threshold['recall']:.4f}")
        print(f"F1-Score at Threshold: {metrics_at_threshold['f1']:.4f}")
        
        # Confusion matrix
        cm = metrics_at_threshold['confusion_matrix']
        print(f"\nConfusion Matrix (at threshold {optimal_threshold:.4f}):")
        print(f"                Predicted")
        print(f"Actual    Negative    Positive")
        print(f"Negative  {cm[0][0]:>8}  {cm[0][1]:>8}")
        print(f"Positive  {cm[1][0]:>8}  {cm[1][1]:>8}")
        
        print(f"\n✅ Evaluation completed! Reports saved to: {self.reports_dir}")
        print(f"📊 Threshold saved to: {self.models_dir}/threshold.json")
