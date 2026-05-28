"""
Metrics Calculation Module
Contains functions to calculate model performance metrics.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
from config import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score.
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        Dict with keys: accuracy, precision, recall, f1
    """
    try:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
        }
    except Exception as e:
        logger.warning(f"Metrics calculation failed: {e}")
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return metrics


def calculate_cv_score(model, X, y, cv=5):
    """
    Calculate cross-validation score.
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of CV folds
    Returns:
        Array of CV scores
    """
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        logger.info(f"CV scores: {scores}, mean={scores.mean():.4f}")
        return scores
    except Exception as e:
        logger.error(f"CV calculation failed: {e}")
        return np.array([])
