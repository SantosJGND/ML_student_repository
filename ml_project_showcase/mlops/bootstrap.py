import sys
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, PROJECT_MODELS, registry_name
from mlops.promotion import transition_to_production

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


def bootstrap_model(model_type):
    import mlflow
    import mlflow.sklearn

    model_path = MODELS_DIR / f"{model_type}.joblib"
    if not model_path.exists():
        logger.warning(f"No .joblib found for {model_type} at {model_path}")
        return False

    model = joblib.load(str(model_path))
    registered_name = registry_name(model_type)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"bootstrap_{model_type}") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("source", "bootstrap")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        result = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
        logger.info(f"{model_type} registered as {registered_name} v{result.version}")

        transition_to_production(registered_name, result.version)
        logger.info(f"{model_type} promoted to Production as v{result.version}")

    return True


def main():
    logger.info("Bootstrapping existing models into MLflow Registry...")
    success = 0
    for mt in PROJECT_MODELS:
        if bootstrap_model(mt):
            success += 1
    logger.info(f"Bootstrapped {success}/{len(PROJECT_MODELS)} models")
    if success == 0:
        logger.warning("No models were bootstrapped. Train models first!")
        sys.exit(1)


if __name__ == "__main__":
    main()
