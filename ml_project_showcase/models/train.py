"""
Model Training Module
Contains functions to train various models on breast cancer data.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from config import FEATURES_DIR, get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(FEATURES_DIR).parent.parent / "models"


def train_logistic(X_train, y_train, params={}):
    """
    Train Logistic Regression model.
    Args:
        X_train: Training features
        y_train: Training target
        params: Dict of hyperparameters (C, solver, max_iter, etc.)
    Returns:
        Trained LogisticRegression model
    """
    default_params = {'max_iter': 1000, 'random_state': 42}
    default_params.update(params)
    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained")
    return model


def train_xgboost(X_train, y_train, params={}):
    """
    Train XGBoost model.
    Args:
        X_train: Training features
        y_train: Training target
        params: Dict of hyperparameters (n_estimators, max_depth, learning_rate, etc.)
    Returns:
        Trained XGBClassifier model
    """
    default_params = {'eval_metric': 'logloss', 'random_state': 42}
    default_params.update(params)
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained")
    return model


def train_random_forest(X_train, y_train, params={}):
    """
    Train Random Forest model.
    Args:
        X_train: Training features
        y_train: Training target
        params: Dict of hyperparameters (n_estimators, max_depth, min_samples_split, etc.)
    Returns:
        Trained RandomForestClassifier model
    """
    default_params = {'random_state': 42}
    default_params.update(params)
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained")
    return model


def train_svm(X_train, y_train, params={}):
    """
    Train SVM model.
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
        params: Dict of hyperparameters (C, kernel, gamma, etc.)
    Returns:
        Trained SVC model
    """
    default_params = {'probability': True, 'random_state': 42}
    default_params.update(params)
    model = SVC(**default_params)
    model.fit(X_train, y_train)
    logger.info("SVM trained")
    return model


def train_all(X_train, y_train, params_dict={}):
    """
    Train all models with optional per-model hyperparameters.
    Args:
        X_train: Training features
        y_train: Training target
        params_dict: Dict mapping model names to hyperparameter dicts
    Returns:
        Dict mapping model names to trained models
    """
    model_funcs = {
        'logistic': train_logistic,
        'xgboost': train_xgboost,
        'random_forest': train_random_forest,
        'svm': train_svm
    }
    models = {}
    for name, func in model_funcs.items():
        params = params_dict.get(name, {})
        models[name] = func(X_train, y_train, params)
    return models


def save_model(model, name):
    """
    Save model to disk using joblib.
    Args:
        model: Trained model to save
        name: Model name (e.g., 'logistic', 'xgboost')
    Returns:
        Path to saved model file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path
