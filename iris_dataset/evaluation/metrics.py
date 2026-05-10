"""
Metrics Module
Contains functions to calculate model evaluation metrics for multiclass classification.
Uses f1_macro for multiclass F1 scoring.
"""


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate evaluation metrics for multiclass classification.

    Args:
        y_true: True labels (0: setosa, 1: versicolor, 2: virginica)
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC AUC OvR)

    Returns:
        Dict with accuracy, f1_macro, precision_macro, recall_macro
    """
    # TODO: Implement metrics calculation with average='macro' for F1
    pass


def calculate_cv_score(model, X, y, cv=5, scoring='accuracy'):
    """
    Calculate cross-validation score.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        CV scores array
    """
    # TODO: Implement cross-validation
    pass
