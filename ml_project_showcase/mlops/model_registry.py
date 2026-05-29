import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI, registry_name, MLFLOW_EXPERIMENT, FEATURES_DIR

logger = get_logger(__name__)

MODELS_DIR = Path(FEATURES_DIR).parent.parent / "models"


def load_production_model(model_type):

    import mlflow
    client = mlflow.tracking.MlflowClient(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    name = registry_name(model_type)
    print(f"Attempting to load {name} from MLflow Registry...")
    stages = ["Production", "Staging"]

    print(client.search_model_versions(f"name='{registry_name(model_type)}'"))
    print(client.get_latest_versions(registry_name(model_type), stages=["Production", "Staging"]))
    versions = client.search_model_versions(f"name='{registry_name(model_type)}'")

    for stage in stages:
        uri = f"models:/{name}/{stage}"
        try:
            model = mlflow.sklearn.load_model(uri)
            logger.info(f"Loaded {name} from {stage}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {name} from {stage}: {e}")
    raise RuntimeError(f"No model found for {name} in Production or Staging")


def get_model_version(model_type):
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    name = registry_name(model_type)
    client = MlflowClient()
    for stage in ["Production", "Staging"]:
        versions = client.search_model_versions(
            f"name='{name}'"
        )
        if versions:
            v = versions[0]
            return {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
    return {"version": None, "stage": None, "run_id": None}
