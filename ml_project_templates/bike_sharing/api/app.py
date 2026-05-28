from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from models.predict import predict_bike_demand
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Bike Sharing ML API")


class PredictRequest(BaseModel):
    Hour: float
    Temperature: float
    Humidity: float
    Wind_speed: float
    Visibility: float
    Dew_point_temperature: float
    Solar_Radiation: float
    Rainfall: float
    Snowfall: float
    Seasons_Spring: int = 0
    Seasons_Summer: int = 0
    Seasons_Winter: int = 0
    Holiday_No_Holiday: int = 0
    Functioning_Day_Yes: int = 1
    model: str = "random_forest"


class PredictResponse(BaseModel):
    predicted_bike_count: float
    model_used: str


class TrainRequest(BaseModel):
    pass


class TrainResponse(BaseModel):
    pass


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pass


@app.get("/health")
def health():
    pass


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    pass
