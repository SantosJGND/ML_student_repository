"""
Gate Evaluation Module
Contains functions to evaluate models and check against gates.
"""
from sklearn.metrics import accuracy_score
from config import get_logger
from evaluation.metrics import calculate_metrics

logger = get_logger(__name__)


def check_gates(metrics, gates):
    """
    Check if metrics pass defined gates.
    Args:
        metrics: Dict of calculated metrics (accuracy, precision, recall, f1)
        gates: Dict of gate thresholds (e.g., {"accuracy": 0.95, "f1": 0.90})
    Returns:
        Tuple: (bool pass/fail, dict of gate check details)
    """
    details = {}
    all_passed = True
    for key, threshold in gates.items():
        value = metrics.get(key, 0.0)
        passed = value >= threshold
        details[key] = {"value": value, "threshold": threshold, "passed": passed}
        if not passed:
            all_passed = False
            logger.warning(f"Gate failed: {key}={value:.4f} < {threshold:.4f}")
    logger.info(f"Gates checked: {'PASSED' if all_passed else 'FAILED'} ({sum(1 for d in details.values() if d['passed'])}/{len(details)} passed)")
    return all_passed, details


def evaluate_and_gate(model, X_test, y_test, gates):
    """
    Evaluate model and check against gates.
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test target
        gates: Dict of gate thresholds
    Returns:
        Tuple: (metrics dict, bool pass/fail)
    """
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    passed, _ = check_gates(metrics, gates)
    return metrics, passed
