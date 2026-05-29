import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from mlops.model_registry import load_production_model, get_model_version
from mlops.monitor import log_prediction, get_metrics
from models.train import (
    train_logistic, train_xgboost,
    train_random_forest, train_svm, save_model
)
from config import get_logger, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, registry_name

logger = get_logger(__name__)

app = FastAPI(title="Breast Cancer ML API")

class PredictRequest(BaseModel):
    Mean_Radius: float = 0.0
    SE_Radius: float = 0.0
    Worst_Radius: float = 0.0
    Mean_Texture: float = 0.0
    SE_Texture: float = 0.0
    Worst_Texture: float = 0.0
    Mean_Perimeter: float = 0.0
    SE_Perimeter: float = 0.0
    Worst_Perimeter: float = 0.0
    Mean_Area: float = 0.0
    SE_Area: float = 0.0
    Worst_Area: float = 0.0
    Mean_Smoothness: float = 0.0
    SE_Smoothness: float = 0.0
    Worst_Smoothness: float = 0.0
    Mean_Compactness: float = 0.0
    SE_Compactness: float = 0.0
    Worst_Compactness: float = 0.0
    Mean_Concavity: float = 0.0
    SE_Concavity: float = 0.0
    Worst_Concavity: float = 0.0
    Mean_ConcavePoints: float = 0.0
    SE_ConcavePoints: float = 0.0
    Worst_ConcavePoints: float = 0.0
    Mean_Symmetry: float = 0.0
    SE_Symmetry: float = 0.0
    Worst_Symmetry: float = 0.0
    Mean_FractalDimension: float = 0.0
    SE_FractalDimension: float = 0.0
    Worst_FractalDimension: float = 0.0
    model: Optional[str] = "logistic"


class TrainRequest(BaseModel):
    model_type: str = "logistic"
    params: Optional[Dict] = {}


class TrainResponse(BaseModel):
    model_type: str
    status: str
    metrics: Optional[Dict] = None
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    message: str


class PredictResponse(BaseModel):
    diagnosis_prediction: int
    malignancy_probability: float
    model_used: str
    model_version: Optional[str] = None


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    features = {
        'Mean_Radius': req.Mean_Radius,
        'SE_Radius': req.SE_Radius,
        'Worst_Radius': req.Worst_Radius,
        'Mean_Texture': req.Mean_Texture,
        'SE_Texture': req.SE_Texture,
        'Worst_Texture': req.Worst_Texture,
        'Mean_Perimeter': req.Mean_Perimeter,
        'SE_Perimeter': req.SE_Perimeter,
        'Worst_Perimeter': req.Worst_Perimeter,
        'Mean_Area': req.Mean_Area,
        'SE_Area': req.SE_Area,
        'Worst_Area': req.Worst_Area,
        'Mean_Smoothness': req.Mean_Smoothness,
        'SE_Smoothness': req.SE_Smoothness,
        'Worst_Smoothness': req.Worst_Smoothness,
        'Mean_Compactness': req.Mean_Compactness,
        'SE_Compactness': req.SE_Compactness,
        'Worst_Compactness': req.Worst_Compactness,
        'Mean_Concavity': req.Mean_Concavity,
        'SE_Concavity': req.SE_Concavity,
        'Worst_Concavity': req.Worst_Concavity,
        'Mean_ConcavePoints': req.Mean_ConcavePoints,
        'SE_ConcavePoints': req.SE_ConcavePoints,
        'Worst_ConcavePoints': req.Worst_ConcavePoints,
        'Mean_Symmetry': req.Mean_Symmetry,
        'SE_Symmetry': req.SE_Symmetry,
        'Worst_Symmetry': req.Worst_Symmetry,
        'Mean_FractalDimension': req.Mean_FractalDimension,
        'SE_FractalDimension': req.SE_FractalDimension,
        'Worst_FractalDimension': req.Worst_FractalDimension,
    }
    import numpy as np
    model_type = req.model or "logistic"
    start = time.time()

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    print(client.search_model_versions(f"name='{registry_name(model_type)}'"))
    print(client.get_latest_versions(registry_name(model_type), stages=["Production", "Staging"]))

    try:
        model = load_production_model(model_type)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    feature_names = list(features.keys())
    X = np.array([[features[f] for f in feature_names]])
    pred = model.predict(X)
    prob = model.predict_proba(X)

    diagnosis = int(pred[0])
    malignancy_prob = float(prob[0][1])
    model_version = get_model_version(model_type)

    latency_ms = (time.time() - start) * 1000
    log_prediction(
        model_version=f"{registry_name(model_type)} v{model_version['version']}",
        prediction=diagnosis,
        latency_ms=latency_ms,
    )

    logger.info(f"Prediction: diagnosis={diagnosis}, prob={malignancy_prob:.3f}, model={model_type}")
    return PredictResponse(
        diagnosis_prediction=diagnosis,
        malignancy_probability=malignancy_prob,
        model_used=model_type,
        model_version=f"v{model_version['version']}" if model_version['version'] else None,
    )


@app.get("/health")
def health():
    versions = {}
    from config import PROJECT_MODELS
    for mt in PROJECT_MODELS:
        v = get_model_version(mt)
        versions[mt] = v
    return {
        "status": "healthy",
        "models": versions,
    }


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    import mlflow
    import mlflow.sklearn

    data_path = Path(__file__).parent.parent / "data" / "processed" / "clean_data.csv"
    df = pd.read_csv(data_path)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_fn = {
        'logistic': train_logistic,
        'xgboost': train_xgboost,
        'random_forest': train_random_forest,
        'svm': train_svm,
    }.get(req.model_type)

    if train_fn is None:
        return TrainResponse(
            model_type=req.model_type,
            status="error",
            message=f"Unknown model type: {req.model_type}"
        )

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    needs_scaling = req.model_type in ("logistic", "svm")
    if needs_scaling:
        clf = train_fn(X_train, y_train, req.params or {})
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        model = pipeline
    else:
        model = train_fn(X_train, y_train, req.params or {})

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=req.model_type) as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", req.model_type)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        registered_name = registry_name(req.model_type)
        result = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
        model_version = result.version

    save_model(model, req.model_type)

    logger.info(f"{req.model_type} trained - acc={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, registered as {registered_name} v{model_version}")

    return TrainResponse(
        model_type=req.model_type,
        status="success",
        metrics=metrics,
        run_id=run_id,
        model_version=str(model_version),
        message=f"Registered as {registered_name} v{model_version}"
    )
