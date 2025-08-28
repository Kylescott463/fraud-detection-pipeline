"""Metrics calculation utilities."""

from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, f1_score, precision_score,
                           recall_score, roc_auc_score)


def calculate_metrics(y_true: Union[List, np.ndarray, pd.Series],
                     y_pred: Union[List, np.ndarray, pd.Series],
                     y_proba: Optional[Union[List, np.ndarray, pd.Series]] = None) -> Dict[str, float]:
    """Calculate comprehensive metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with calculated metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add ROC AUC if probabilities are provided
    if y_proba is not None:
        y_proba = np.array(y_proba)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = 0.0
    
    return metrics


def calculate_classification_report(y_true: Union[List, np.ndarray, pd.Series],
                                  y_pred: Union[List, np.ndarray, pd.Series]) -> Dict:
    """Calculate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with detailed classification metrics
    """
    return classification_report(y_true, y_pred, output_dict=True)


def calculate_confusion_matrix(y_true: Union[List, np.ndarray, pd.Series],
                             y_pred: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
    """Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> Dict[str, str]:
    """Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        decimal_places: Number of decimal places to display
        
    Returns:
        Dictionary with formatted metric strings
    """
    return {k: f"{v:.{decimal_places}f}" for k, v in metrics.items()}
