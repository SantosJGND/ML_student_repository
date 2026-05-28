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
    # TODO: Implement Logistic Regression training
    # Hint: Use LogisticRegression(**params) and fit on X_train, y_train
    pass


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
    # TODO: Implement XGBoost training
    # Hint: Use XGBClassifier(eval_metric='logloss', **params) and fit on X_train, y_train
    pass


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
    # TODO: Implement Random Forest training
    # Hint: Use RandomForestClassifier(**params) and fit on X_train, y_train
    pass


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
    # TODO: Implement SVM training
    # Hint: Use SVC(probability=True, **params) and fit on X_train, y_train
    pass


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
    # TODO: Implement training all models
    # Hint: Define model mapping, iterate through, train each with params_dict.get(name, {})
    pass


def save_model(model, name):
    """
    Save model to disk using joblib.
    Args:
        model: Trained model to save
        name: Model name (e.g., 'logistic', 'xgboost')
    Returns:
        Path to saved model file
    """
    # TODO: Implement model saving
    # Hint: Create path using FEATURES_DIR.parent / "models" / f"{name}.joblib"
    pass
