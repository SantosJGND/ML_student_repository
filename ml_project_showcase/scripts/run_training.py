#!/usr/bin/env python3
"""
Model Training Script
Trains specified model type with optional hyperparameters.
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train breast cancer model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['logistic', 'xgboost', 'random_forest', 'svm'],
                        help='Model type to train')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON string of hyperparameters')
    args = parser.parse_args()

    # TODO: Load data from args.data_path
    # df = ...

    # TODO: Parse params from JSON
    # params = json.loads(args.params)

    # TODO: Split data into X_train, X_test, y_train, y_test
    # ...

    # TODO: Train model based on args.model_type
    # if args.model_type == 'logistic':
    #     model = train_logistic(X_train, y_train, params)
    # elif args.model_type == 'xgboost':
    #     model = train_xgboost(X_train, y_train, params)
    # ...

    # TODO: Evaluate model
    # metrics = calculate_metrics(y_test, model.predict(X_test))

    # TODO: Log to MLflow
    # mlflow.log_metrics(metrics)

    # TODO: Save model
    # save_model(model, args.model_type)

    logger.info(f"Training script completed for {args.model_type} (TODO: implement logic)")


if __name__ == "__main__":
    main()
