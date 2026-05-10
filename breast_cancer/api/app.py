from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict_diagnosis, predict_malignancy_prob
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Breast Cancer ML API")


class PredictRequest(BaseModel):
    # 30 features from the breast cancer dataset
    pass


class PredictResponse(BaseModel):
    #
    pass

class TrainRequest(BaseModel):
    # 
    pass

class TrainResponse(BaseModel):
    # 
    pass

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # TODO: Extract features from req into a dict
    # features = {
    #     'Mean_Radius': req.Mean_Radius,
    #     'SE_Radius': req.SE_Radius,
    #     ...
    # }
    
    # TODO: Call predict_diagnosis() with req.model
    # diagnosis = predict_diagnosis(req.model, features)
    
    # TODO: Call predict_malignancy_prob() with req.model
    # prob = predict_malignancy_prob(req.model, features)
    
    # TODO: Handle None cases (return 0 for diagnosis, 0.0 for prob)
    # if diagnosis is None:
    #     diagnosis = 0
    # if prob is None:
    #     prob = 0.0
    
    logger.info(f"Prediction: diagnosis={diagnosis}, prob={prob:.3f}, model={req.model}")
    return PredictResponse(
        diagnosis_prediction=diagnosis,
        malignancy_probability=prob,
        model_used=req.model
    )


@app.get("/health")
def health():
    # TODO: Return {"status": "healthy"}
    pass


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    # TODO: Implement training logic (optional for demo)
    pass
