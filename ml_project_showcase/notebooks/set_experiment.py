from pathlib import Path
import mlflow
import sys

# Add project root to path so we can import project modules
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_DIR = Path("..") / "mlflow_artifacts"
MLFLOW_TRACKING_URI = str(MLFLOW_DIR.absolute())
MLFLOW_EXPERIMENT = "breast_cancer_ml"
print(f"Project root: {PROJECT_ROOT}")
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("breast_cancer_06_registry")
