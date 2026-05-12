"""
Model Prediction Module
Contains functions to load models and make predictions.
"""
import joblib
from pathlib import Path
from config import FEATURES_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(FEATURES_DIR).parent.parent / "models"


def load_model(name):
    """
    Load model from MLflow Registry or local file.
    Args:
        name: Model name (e.g., 'logistic', 'xgboost')
    Returns:
        Loaded model or None if not found
    """
    # Try MLflow first (if available)
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{name}/latest"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded {name} from MLflow registry")
        return model
    except ImportError:
        logger.warning("mlflow not installed, skipping MLflow load")
    except Exception as e:
        logger.warning(f"MLflow load failed: {e}")

    # Fallback to joblib
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        logger.error(f"Model {name} not found at {path}")
        return None
    return joblib.load(path)


def predict(model, X):
    """
    Make predictions using loaded model.
    Args:
        model: Trained model
        X: Features to predict on (numpy array or dict)
    Returns:
        y_pred: Predicted labels
    """
    import numpy as np
    if isinstance(X, dict):
        # Convert dict to array in correct feature order
        feature_names = ['Mean_Radius', 'SE_Radius', 'Worst_Radius',
                        'Mean_Texture', 'SE_Texture', 'Worst_Texture',
                        'Mean_Perimeter', 'SE_Perimeter', 'Worst_Perimeter',
                        'Mean_Area', 'SE_Area', 'Worst_Area',
                        'Mean_Smoothness', 'SE_Smoothness', 'Worst_Smoothness',
                        'Mean_Compactness', 'SE_Compactness', 'Worst_Compactness',
                        'Mean_Concavity', 'SE_Concavity', 'Worst_Concavity',
                        'Mean_ConcavePoints', 'SE_ConcavePoints', 'Worst_ConcavePoints',
                        'Mean_Symmetry', 'SE_Symmetry', 'Worst_Symmetry',
                        'Mean_FractalDimension', 'SE_FractalDimension', 'Worst_FractalDimension']
        X = np.array([[X.get(f, 0.0) for f in feature_names]])
    return model.predict(X)


def predict_proba(model, X):
    """
    Predict probabilities using loaded model.
    Args:
        model: Trained model with predict_proba support
        X: Features to predict on
    Returns:
        y_proba: Predicted probabilities (shape: [n_samples, n_classes])
    """
    import numpy as np
    if isinstance(X, dict):
        feature_names = ['Mean_Radius', 'SE_Radius', 'Worst_Radius',
                        'Mean_Texture', 'SE_Texture', 'Worst_Texture',
                        'Mean_Perimeter', 'SE_Perimeter', 'Worst_Perimeter',
                        'Mean_Area', 'SE_Area', 'Worst_Area',
                        'Mean_Smoothness', 'SE_Smoothness', 'Worst_Smoothness',
                        'Mean_Compactness', 'SE_Compactness', 'Worst_Compactness',
                        'Mean_Concavity', 'SE_Concavity', 'Worst_Concavity',
                        'Mean_ConcavePoints', 'SE_ConcavePoints', 'Worst_ConcavePoints',
                        'Mean_Symmetry', 'SE_Symmetry', 'Worst_Symmetry',
                        'Mean_FractalDimension', 'SE_FractalDimension', 'Worst_FractalDimension']
        X = np.array([[X.get(f, 0.0) for f in feature_names]])
    return model.predict_proba(X)


def predict_diagnosis(model_name, features_dict):
    """
    Predict breast cancer diagnosis (0=benign, 1=malignant).
    Args:
        model_name: Name of model to use
        features_dict: Dict with 30 feature values
    Returns:
        Predicted diagnosis (0 or 1), or None if model fails
    """
    model = load_model(model_name)
    if model is None:
        return None
    try:
        pred = predict(model, features_dict)
        return int(pred[0])
    except Exception as e:
        logger.warning(f"Prediction failed: {e}")
        return None


def predict_malignancy_prob(model_name, features_dict):
    """
    Predict probability of malignant diagnosis.
    Args:
        model_name: Name of model to use (should support predict_proba)
        features_dict: Dict with 30 feature values
    Returns:
        Probability of malignancy (class 1), or None if model fails
    """
    model = load_model(model_name)
    if model is None:
        return None
    try:
        proba = predict_proba(model, features_dict)
        return float(proba[0][1])  # Probability of class 1 (malignant)
    except Exception as e:
        logger.warning(f"Probability prediction failed: {e}")
        return None
