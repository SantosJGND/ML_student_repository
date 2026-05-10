import joblib
from pathlib import Path
from config import FEATURES_DIR, get_logger

logger = get_logger(__name__)

MODELS_DIR = Path(FEATURES_DIR).parent / "models"


def load_model(name):
    pass


def predict(model, X):
    pass


def predict_bike_demand(model_name, features_dict):
    pass
