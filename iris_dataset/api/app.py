from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict_species, predict_species_proba
from models.train import (
    train_logistic_regression, train_xgboost,
    train_random_forest, train_svm, save_model
)
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Iris Classification ML API")


class PredictRequest(BaseModel):
    # 4 features from the iris dataset
    sepal_length: float = 0.0
    sepal_width: float = 0.0
    petal_length: float = 0.0
    petal_width: float = 0.0
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
    species_prediction: int  # 0=setosa, 1=versicolor, 2=virginica
    probabilities: Dict[str, float]  # {"setosa": 0.1, "versicolor": 0.3, "virginica": 0.6}
    model_used: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # TODO: Extract features from req into a dict
    # features = {
    #     'sepal_length': req.sepal_length,
    #     'sepal_width': req.sepal_width,
    #     'petal_length': req.petal_length,
    #     'petal_width': req.petal_width
    # }
    
    # TODO: Call predict_species() with req.model
    # species = predict_species(req.model, features)
    
    # TODO: Call predict_species_proba() with req.model
    # proba = predict_species_proba(req.model, features)
    # probs_dict = {'setosa': float(proba[0][0]), ...}
    
    # TODO: Handle None cases
    # if species is None: species = 0
    # if probs_dict is None: probs_dict = {"setosa": 0.33, ...}
    
    logger.info(f"Prediction: species={species}, model={req.model}")
    return PredictResponse(
        species_prediction=species,
        probabilities=probs_dict,
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
    # X = df.drop('class', axis=1)
    # y = df['class']
    
    # TODO: Split data (hardcoded test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(...)
    
    # TODO: Train model based on req.model_type
    # if req.model_type == 'logistic':
    #     model = train_logistic_regression(X_train, y_train, req.params)
    # elif req.model_type == 'xgboost':
    #     model = train_xgboost(X_train, y_train, req.params)
    # ...
    
    # TODO: Evaluate on test set
    # y_pred = model.predict(X_test)
    # metrics = {'accuracy': ..., 'f1_macro': ...}
    
    # TODO: Save model (backend handles path)
    # save_model(model, req.model_type)
    
    logger.info(f"Model {req.model_type} trained")
    return TrainResponse(
        model_type=req.model_type,
        status="success",
        message=f"Model saved to models/{req.model_type}.joblib"
    )
