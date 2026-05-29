#!/usr/bin/env python3
"""
Model Registration Script
Registers a model run in the MLflow Model Registry.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--model-uri", type=str, required=True,
                        help="MLflow model URI to register (e.g., runs:/<run_id>/model)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name for registered model (e.g., breast_cancer_pipeline)")
    args = parser.parse_args()

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info(f"Registering {args.model_uri} as '{args.model_name}'")
    result = mlflow.register_model(args.model_uri, args.model_name)

    print(f"Model registered: '{args.model_name}' version {result.version}")
    print(f"  Run ID: {result.run_id}")
    print(f"  Stage: {result.current_stage}")
    logger.info(f"Registration completed: version {result.version}")


if __name__ == "__main__":
    main()
