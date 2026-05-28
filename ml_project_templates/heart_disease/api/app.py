from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict_diagnosis, predict_probability
from models.train import (
    train_logistic_regression, train_xgboost,
    train_random_forest, train_svm, save_model
)
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Heart Disease ML API")


class PredictRequest(BaseModel):
    # 13 features from the heart disease dataset
    age: float = 0.0
    sex: float = 0.0
    cp: float = 0.0
    trestbps: float = 0.0
    chol: float = 0.0
    fbs: float = 0.0
    restecg: float = 0.0
    thalach: float = 0.0
    exang: float = 0.0
    oldpeak: float = 0.0
    slope: float = 0.0
    ca: float = 0.0
    thal: float = 0.0
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
    probability: float
    model_used: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # TODO: Extract features from req into a dict
    # features = {
    #     'age': req.age,
    #     'sex': req.sex,
    #     ...
    # }
    
    # TODO: Call predict_diagnosis() with req.model
    # diagnosis = predict_diagnosis(req.model, features)
    
    # TODO: Call predict_probability() with req.model
    # prob = predict_probability(req.model, features)
    
    # TODO: Handle None cases (return 0 for diagnosis, 0.0 for prob)
    # if diagnosis is None:
    #     diagnosis = 0
    # if prob is None:
    #     prob = 0.0
    
    logger.info(f"Prediction: diagnosis={diagnosis}, prob={prob:.3f}, model={req.model}")
    return PredictResponse(
        diagnosis_prediction=diagnosis,
        probability=prob,
        model_used=req.model
    )


@app.get("/health")
def health():
    # TODO: Return {"status": "healthy"}
    pass


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    # Hardcoded data path (not exposed to user)
    data_path = Path(__file__).parent.parent / "data" / "processed" / "clean_data.csv"

    # TODO: Load data from data_path
    # df = pd.read_csv(data_path)
    # X = df.drop('target', axis=1)
    # y = df['target']
    
    # TODO: Split data (hardcoded test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    
    # TODO: Train model based on req.model_type
    # if req.model_type == 'logistic':
    #     model = train_logistic_regression(X_train, y_train, req.params)
    # elif req.model_type == 'xgboost':
    #     model = train_xgboost(X_train, y_train, req.params)
    # ...
    
    # TODO: Evaluate on test set
    # y_pred = model.predict(X_test)
    # metrics = {
    #     'accuracy': float(accuracy_score(y_test, y_pred)),
    #     'f1': float(f1_score(y_test, y_pred))
    # }
    
    # TODO: Save model (backend handles path)
    # save_model(model, req.model_type)
    
    logger.info(f"Model {req.model_type} trained")
    return TrainResponse(
        model_type=req.model_type,
        status="success",
        message=f"Model saved to models/{req.model_type}.joblib"
    )
