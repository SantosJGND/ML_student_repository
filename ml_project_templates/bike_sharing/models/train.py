from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import joblib
from pathlib import Path
from config import FEATURES_DIR, get_logger

logger = get_logger(__name__)


def train_linear_regression(X_train, y_train, params={}):
    pass


def train_xgboost(X_train, y_train, params={}):
    pass


def train_random_forest(X_train, y_train, params={}):
    pass


def train_svr(X_train, y_train, params={}):
    pass


def train_all(X_train, y_train, params_dict={}):
    pass


def save_model(model, name):
    pass
