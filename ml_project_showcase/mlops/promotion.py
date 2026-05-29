import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI

logger = get_logger(__name__)


def _get_client():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def transition_to_staging(registered_name, run_id):

    client = _get_client()

    versions = client.search_model_versions(
        f"name='{registered_name}'"
    )

    # ONLY select existing versions
    candidates = [
        v for v in versions if v.run_id == run_id
    ]

    if not candidates:
        raise RuntimeError(
            f"No registered model found for run_id={run_id}. "
            "Do NOT register models in promotion step."
        )

    v = sorted(
        candidates,
        key=lambda x: int(x.version),
        reverse=True
    )[0]

    client.transition_model_version_stage(
        name=registered_name,
        version=v.version,
        stage="Staging"
    )

    logger.info(f"{registered_name} v{v.version} → Staging")

    return v.version


def transition_to_production(registered_name, version):
    client = _get_client()
    client.transition_model_version_stage(
        name=registered_name,
        version=version,
        stage="Production"
    )
    logger.info(f"{registered_name} v{version} → Production")
    for v in client.search_model_versions(f"name='{registered_name}'"):
        if v.version != version and v.current_stage == "Production":
            client.transition_model_version_stage(
                name=registered_name,
                version=v.version,
                stage="Archived"
            )
            logger.info(f"{registered_name} v{v.version} → Archived")


def archive_previous(registered_name, keep_version=None):
    client = _get_client()
    for v in client.search_model_versions(f"name='{registered_name}'"):
        if v.current_stage == "Production" and (keep_version is None or v.version != keep_version):
            client.transition_model_version_stage(
                name=registered_name,
                version=v.version,
                stage="Archived"
            )
            logger.info(f"{registered_name} v{v.version} → Archived")
