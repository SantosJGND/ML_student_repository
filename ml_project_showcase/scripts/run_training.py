#!/usr/bin/env python3
"""
Model Training Script
Trains a sklearn Pipeline (preprocessing + estimator) with optional hyperparameters,
logs to MLflow, and saves to disk.
"""
import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from models.train import train_logistic, train_xgboost, train_random_forest, train_svm, save_model
from evaluation.metrics import calculate_metrics

logger = get_logger(__name__)

MODEL_FUNCS = {
    "logistic": train_logistic,
    "xgboost": train_xgboost,
    "random_forest": train_random_forest,
    "svm": train_svm,
}


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=list(MODEL_FUNCS.keys()),
                        help="Model type to train")
    parser.add_argument("--params", type=str, default="{}",
                        help="JSON string of hyperparameters")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set proportion")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    params = json.loads(args.params)
    train_fn = MODEL_FUNCS[args.model_type]

    needs_scaling = args.model_type in ("logistic", "svm")
    if needs_scaling:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", train_fn(X_train, y_train, params)),
        ])
        pipeline.fit(X_train, y_train)
        model = pipeline
    else:
        model = train_fn(X_train, y_train, params)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print(f"\n{args.model_type} results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_path = save_model(model, args.model_type)
    print(f"Model saved: {save_path}")

    if not args.no_mlflow:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=args.model_type) as run:
            mlflow.log_param("model_type", args.model_type)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
            logger.info(f"MLflow run: {run.info.run_id}")
            print(f"MLflow run_id: {run.info.run_id}")

    logger.info(f"Training completed for {args.model_type}")


if __name__ == "__main__":
    main()
