import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, PROJECT_MODELS, registry_name
from models.train import train_logistic, train_xgboost, train_random_forest, train_svm, save_model
from evaluation.metrics import calculate_metrics
from mlops.gate import evaluate_and_promote

logger = get_logger(__name__)

MODEL_FUNCS = {
    "logistic": train_logistic,
    "xgboost": train_xgboost,
    "random_forest": train_random_forest,
    "svm": train_svm,
}

NEEDS_SCALING = {"logistic", "svm"}


def train_model(model_type, X_train, y_train, params=None):
    train_fn = MODEL_FUNCS[model_type]
    params = params or {}
    if model_type in NEEDS_SCALING:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", train_fn(X_train, y_train, params)),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline
    return train_fn(X_train, y_train, params)


def train_and_register(model_type, X_train, y_train, X_test, y_test, params=None):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    registered_name = registry_name(model_type)
    model = train_model(model_type, X_train, y_train, params)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    signature = infer_signature(X_test, y_pred)

    with mlflow.start_run(run_name=model_type) as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", 0.2)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        
        mlflow.sklearn.log_model(sk_model=model,
                                 name="model",
                                 signature=signature,
                                 registered_model_name=registered_name,
                                 serialization_format="pickle"
        )
        # 🔍 HARD CHECK (critical)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)


        print("ARTIFACTS:", artifacts)
        model_path= save_model(model, model_type)
        mlflow.log_artifact(model_path, "backup")
        #model_uri = f"runs:/{run_id}/model"
        
        mlflow.end_run()   # forces flush

    
    logger.info(f"{model_type} logged an")

    logger.info(f"{model_type} saved to disk (backup)")

    
    return {"model_type": model_type, "metrics": metrics, "run_id": run_id}


def register_model(model_type, run_id):
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    registered_name = registry_name(model_type)
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=registered_name)
    logger.info(f"{model_type} registered as {registered_name}")
    return registered_name


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full training pipeline for breast_cancer")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-types", type=str, default="all",
                        help="Comma-separated list or 'all'")
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = args.data_path or (Path(__file__).parent.parent / "data" / "processed" / "clean_data.csv")
    params_dict = json.loads(args.params)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model_types = PROJECT_MODELS if args.model_types == "all" else args.model_types.split(",")

    results = []
    for mt in model_types:
        mt = mt.strip()
        if mt not in MODEL_FUNCS:
            logger.warning(f"Unknown model type: {mt}, skipping")
            continue
        logger.info(f"Training {mt}...")
        params = params_dict.get(mt, {})
        result = train_and_register(mt, X_train, y_train, X_test, y_test, params)
        results.append(result)


    print("\nSummary:")
    for r in results:
        register_model(r["model_type"], r["run_id"])
        passed, details = evaluate_and_promote(r["model_type"], r["metrics"], r["run_id"])
        r["passed"] = passed
        r["details"] = details

        print(f"  {'✓' if r['passed'] else '✗'} {r['model_type']:15s} acc={r['metrics']['accuracy']:.4f}  f1={r['metrics']['f1']:.4f}  run_id={r['run_id'][:8]}")

    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
