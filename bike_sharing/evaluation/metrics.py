from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
from config import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true, y_pred):
    pass


def calculate_cv_score(model, X, y, cv=5):
    pass
