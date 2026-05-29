#!/usr/bin/env python3
"""
Gate Evaluation Script
Evaluates a model against CLI-configurable gate thresholds.
Loads model from MLflow registry or local joblib file.
"""
import argparse
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI
from evaluation.metrics import calculate_metrics
from evaluation.gate import check_gates

logger = get_logger(__name__)


def load_model_from_uri(uri):
    """Load model from MLflow URI or local joblib path."""
    import joblib
    if uri.startswith("models:"):
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(uri)
        logger.info(f"Loaded model from MLflow: {uri}")
        return model
    else:
        model = joblib.load(uri)
        logger.info(f"Loaded model from joblib: {uri}")
        return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model gates")
    parser.add_argument("--model-uri", type=str, required=True,
                        help="MLflow model URI (e.g., models:/breast_cancer_pipeline/1) or path to .joblib")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test data CSV")
    parser.add_argument("--gates", type=str,
                        default='{"accuracy": 0.95, "f1": 0.90}',
                        help="JSON string of gate thresholds")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_uri}")
    model = load_model_from_uri(args.model_uri)

    logger.info(f"Loading test data from {args.test_data}")
    df = pd.read_csv(args.test_data)
    X_test = df.drop(columns=["Diagnosis"])
    y_test = df["Diagnosis"]

    gates = json.loads(args.gates)
    logger.info(f"Gate thresholds: {gates}")

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print("\nModel metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    passed, details = check_gates(metrics, gates)

    print(f"\nGate check: {'✓ PASSED' if passed else '✗ FAILED'}")
    for key, detail in details.items():
        status = "✓" if detail["passed"] else "✗"
        print(f"  {status} {key}: {detail['value']:.4f} >= {detail['threshold']:.4f}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
