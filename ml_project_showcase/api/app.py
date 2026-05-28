from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict_diagnosis, predict_malignancy_prob
from models.train import (
    train_logistic, train_xgboost,
    train_random_forest, train_svm, save_model
)
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Breast Cancer ML API")

class PredictRequest(BaseModel):
    # 30 features from the breast cancer dataset
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
    message: str


class PredictResponse(BaseModel):
    diagnosis_prediction: int
    malignancy_probability: float
    model_used: str


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
    model_name = req.model or "logistic"
    diagnosis = predict_diagnosis(model_name, features)
    prob = predict_malignancy_prob(model_name, features)
    if diagnosis is None:
        diagnosis = 0
    if prob is None:
        prob = 0.0
    logger.info(f"Prediction: diagnosis={diagnosis}, prob={prob:.3f}, model={model_name}")
    return PredictResponse(
        diagnosis_prediction=diagnosis,
        malignancy_probability=prob,
        model_used=model_name
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    # Hardcoded data path (not exposed to user)
    data_path = Path(__file__).parent.parent / "data" / "processed" / "clean_data.csv"

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Split data (hardcoded test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model based on type
    model = None
    if req.model_type == 'logistic':
        model = train_logistic(X_train, y_train, req.params)
    elif req.model_type == 'xgboost':
        model = train_xgboost(X_train, y_train, req.params)
    elif req.model_type == 'random_forest':
        model = train_random_forest(X_train, y_train, req.params)
    elif req.model_type == 'svm':
        model = train_svm(X_train, y_train, req.params)
    else:
        return TrainResponse(
            model_type=req.model_type,
            status="error",
            message=f"Unknown model type: {req.model_type}"
        )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred))
    }

    # Save model (backend handles path)
    save_model(model, req.model_type)

    logger.info(f"Model {req.model_type} trained - accuracy: {metrics['accuracy']:.3f}, f1: {metrics['f1']:.3f}")

    return TrainResponse(
        model_type=req.model_type,
        status="success",
        metrics=metrics,
        message=f"Model saved to models/{req.model_type}.joblib"
    )
