import os
from pathlib import Path
import logging

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MLFLOW_DIR = Path(__file__).parent / "mlflow_artifacts"
MLFLOW_TRACKING_URI = str(MLFLOW_DIR.absolute())
MLFLOW_EXPERIMENT = "breast_cancer_ml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name):
    return logging.getLogger(name)
