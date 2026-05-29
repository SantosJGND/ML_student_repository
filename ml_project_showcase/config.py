import os
from pathlib import Path
import logging

PROJECT_NAME = "breast_cancer"
RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", PROJECT_NAME)
MLRUNS_DIR = os.getenv("MLRUNS_DIR", "./mlruns")

PROJECT_MODELS = ["logistic", "xgboost", "random_forest", "svm"]

def registry_name(model_type):
    return f"{PROJECT_NAME}_{model_type}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name):
    return logging.getLogger(name)
