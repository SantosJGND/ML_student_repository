"""
Model Prediction Module
Contains functions to load models and make predictions.
"""
import joblib
from pathlib import Path
from config import FEATURES_DIR, get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(FEATURES_DIR).parent / "models"


def load_model(name):
    """
    Load model from MLflow Registry or local file.
    Args:
        name: Model name (e.g., 'logistic', 'xgboost')
    Returns:
        Loaded model or None if not found
    """
    # TODO: Implement model loading
    # Hint: Try MLflow first (if available), fallback to joblib.load()
    # Example:
    # try:
    #     import mlflow
    #     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #     model = mlflow.pyfunc.load_model(f"models:/{name}/latest")
    #     return model
    # except:
    #     pass
    # path = MODELS_DIR / f"{name}.joblib"
    # if path.exists():
    #     return joblib.load(path)
    # return None
    pass


def predict(model, X):
    """
    Make predictions using loaded model.
    Args:
        model: Trained model
        X: Features to predict on
    Returns:
        y_pred: Predicted labels
    """
    # TODO: Implement prediction
    # Hint: Use model.predict(X)
    pass


def predict_proba(model, X):
    """
    Predict probabilities using loaded model.
    Args:
        model: Trained model with predict_proba support
        X: Features to predict on
    Returns:
        y_proba: Predicted probabilities (shape: [n_samples, n_classes])
    """
    # TODO: Implement probability prediction
    # Hint: Use model.predict_proba(X)
    pass


def predict_diagnosis(model_name, features_dict):
    """
    Predict breast cancer diagnosis (0=benign, 1=malignant).
    Args:
        model_name: Name of model to use
        features_dict: Dict with 30 feature values
    Returns:
        Predicted diagnosis (0 or 1), or None if model fails
    """
    # TODO: Implement diagnosis prediction
    # Hint: Load model with load_model(), call predict() with features_dict
    pass


def predict_malignancy_prob(model_name, features_dict):
    """
    Predict probability of malignant diagnosis.
    Args:
        model_name: Name of model to use (should support predict_proba)
        features_dict: Dict with 30 feature values
    Returns:
        Probability of malignancy (class 1), or None if model fails
    """
    # TODO: Implement malignancy probability prediction
    # Hint: Load model with load_model(), call predict_proba(), return proba[0][1]
    pass
