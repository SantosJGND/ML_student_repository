#!/usr/bin/env python3
"""
Model Training Script
Trains iris classification model with specified algorithm and hyperparameters.
Multiclass classification (3 species).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train iris classification model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['logistic', 'xgboost', 'random_forest', 'svm'],
                        help='Model type to train')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON string of hyperparameters')
    args = parser.parse_args()

    # TODO: Load data from args.data_path
    # TODO: Parse args.params as JSON for model hyperparameters
    # TODO: For XGBoost, use eval_metric='mlogloss' for multiclass
    # TODO: For SVM, use decision_function_shape='ovr' for multiclass
    # TODO: Train model using models/train.py based on args.model_type
    # TODO: Log model to MLflow with params and metrics
    # TODO: Save model artifact

    logger.info("Training script completed (TODO: implement logic)")


if __name__ == "__main__":
    main()
