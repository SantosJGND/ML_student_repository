import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI
from mlops.monitor import LOG_FILE

logger = get_logger(__name__)


def clean():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    deleted_models = []
    for rm in client.search_registered_models():
        name = rm.name
        for v in rm.latest_versions:
            client.delete_model_version(name, v.version)
        client.delete_registered_model(name)
        deleted_models.append(name)

    deleted_experiments = []
    for exp in client.search_experiments(view_type=mlflow.entities.ViewType.ALL):
        if exp.experiment_id != "0":
            client.delete_experiment(exp.experiment_id)
            deleted_experiments.append(exp.name)

    cleaned_log = False
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        cleaned_log = True

    print("Cleaned MLflow state:")
    print(f"  Registered models deleted: {len(deleted_models)}")
    for n in deleted_models:
        print(f"    - {n}")
    print(f"  Experiments deleted: {len(deleted_experiments)}")
    for n in deleted_experiments:
        print(f"    - {n}")
    print(f"  Monitoring log cleaned: {cleaned_log}")


if __name__ == "__main__":
    clean()
