from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from models.predict import predict_income, predict_high_income_prob
from models.train import (
    train_logistic_regression, train_xgboost,
    train_random_forest, train_svm, save_model
)
from config import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Income Prediction ML API")


class PredictRequest(BaseModel):
    # Numeric features
    age: float = 0.0
    fnlwgt: float = 0.0
    education_num: float = 0.0
    capital_gain: float = 0.0
    capital_loss: float = 0.0
    hours_per_week: float = 0.0

    # One-hot encoded: workclass
    workclass_Federal_gov: bool = False
    workclass_Local_gov: bool = False
    workclass_Never_worked: bool = False
    workclass_Private: bool = False
    workclass_Self_emp_inc: bool = False
    workclass_Self_emp_not_inc: bool = False
    workclass_State_gov: bool = False
    workclass_Without_pay: bool = False

    # One-hot encoded: education
    education_11th: bool = False
    education_12th: bool = False
    education_1st_4th: bool = False
    education_5th_6th: bool = False
    education_7th_8th: bool = False
    education_9th: bool = False
    education_Assoc_acdm: bool = False
    education_Assoc_voc: bool = False
    education_Bachelors: bool = False
    education_Doctorate: bool = False
    education_HS_grad: bool = False
    education_Masters: bool = False
    education_Preschool: bool = False
    education_Prof_school: bool = False
    education_Some_college: bool = False

    # One-hot encoded: marital-status
    marital_status_Married_AF_spouse: bool = False
    marital_status_Married_civ_spouse: bool = False
    marital_status_Married_spouse_absent: bool = False
    marital_status_Never_married: bool = False
    marital_status_Separated: bool = False
    marital_status_Widowed: bool = False

    # One-hot encoded: occupation
    occupation_Adm_clerical: bool = False
    occupation_Armed_Forces: bool = False
    occupation_Craft_repair: bool = False
    occupation_Exec_managerial: bool = False
    occupation_Farming_fishing: bool = False
    occupation_Handlers_cleaners: bool = False
    occupation_Machine_op_inspct: bool = False
    occupation_Other_service: bool = False
    occupation_Priv_house_serv: bool = False
    occupation_Prof_specialty: bool = False
    occupation_Protective_serv: bool = False
    occupation_Sales: bool = False
    occupation_Tech_support: bool = False
    occupation_Transport_moving: bool = False

    # One-hot encoded: relationship
    relationship_Not_in_family: bool = False
    relationship_Other_relative: bool = False
    relationship_Own_child: bool = False
    relationship_Unmarried: bool = False
    relationship_Wife: bool = False

    # One-hot encoded: race
    race_Asian_Pac_Islander: bool = False
    race_Black: bool = False
    race_Other: bool = False
    race_White: bool = False

    # One-hot encoded: sex
    sex_Male: bool = False
    sex_Female: bool = True

    # One-hot encoded: native-country (simplified - major countries)
    native_country_Cambodia: bool = False
    native_country_Canada: bool = False
    native_country_China: bool = False
    native_country_Columbia: bool = False
    native_country_Cuba: bool = False
    native_country_Dominican_Republic: bool = False
    native_country_Ecuador: bool = False
    native_country_El_Salvador: bool = False
    native_country_England: bool = False
    native_country_France: bool = False
    native_country_Germany: bool = False
    native_country_Guatemala: bool = False
    native_country_Haiti: bool = False
    native_country_Holand_Netherlands: bool = False
    native_country_Honduras: bool = False
    native_country_Hong: bool = False
    native_country_Hungary: bool = False
    native_country_India: bool = False
    native_country_Iran: bool = False
    native_country_Ireland: bool = False
    native_country_Italy: bool = False
    native_country_Jamaica: bool = False
    native_country_Japan: bool = False
    native_country_Laos: bool = False
    native_country_Mexico: bool = False
    native_country_Nicaragua: bool = False
    native_country_Outlying_US_Guam_USVI_etc: bool = False
    native_country_Peru: bool = False
    native_country_Philippines: bool = False
    native_country_Poland: bool = False
    native_country_Portugal: bool = False
    native_country_Puerto_Rico: bool = False
    native_country_Scotland: bool = False
    native_country_South: bool = False
    native_country_Taiwan: bool = False
    native_country_Thailand: bool = False
    native_country_Trinadad_Tobago: bool = False
    native_country_United_States: bool = True
    native_country_Vietnam: bool = False
    native_country_Yugoslavia: bool = False

    model: Optional[str] = "logistic"


class TrainRequest(BaseModel):
    model_type: str = "logistic"
    params: Optional[Dict] = {}
    data_path: Optional[str] = "data/processed/clean_train.csv"


class TrainResponse(BaseModel):
    model_type: str
    status: str
    metrics: Optional[Dict] = None
    message: str


class PredictResponse(BaseModel):
    income_prediction: int
    probability: float
    model_used: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # TODO: Extract features from req into a dict
    # Hint: Iterate through req attributes, convert bool to float (1.0/0.0)

    # TODO: Call predict_income() with req.model
    # income = predict_income(req.model, features)

    # TODO: Call predict_high_income_prob() with req.model
    # prob = predict_high_income_prob(req.model, features)

    # TODO: Handle None cases (return 0 for income, 0.0 for prob)

    logger.info(f"Prediction: income={income}, prob={prob:.3f}, model={req.model}")
    return PredictResponse(
        income_prediction=income,
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
    data_path = Path(__file__).parent.parent / "data" / "processed" / "clean_train.csv"

    # TODO: Load data from data_path
    # df = pd.read_csv(data_path)
    # X = df.drop('income', axis=1)
    # y = df['income']

    # TODO: Split data (hardcoded test_size=0.2)
    # X_train, X_test, y_train, y_test = train_test_split(...)

    # TODO: Train model based on req.model_type
    # if req.model_type == 'logistic':
    #     model = train_logistic_regression(X_train, y_train, req.params)

    # TODO: Evaluate on test set
    # y_pred = model.predict(X_test)
    # metrics = {...}

    # TODO: Save model (backend handles path)
    # save_model(model, req.model_type)

    logger.info(f"Model {req.model_type} trained")
    return TrainResponse(
        model_type=req.model_type,
        status="success",
        message=f"Model saved to models/{req.model_type}.joblib"
    )
