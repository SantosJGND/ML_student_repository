"""
Model Training Module
Contains functions to train different ML models for heart disease prediction.
"""


def train_logistic_regression(X_train, y_train, params=None):
    """
    Train a logistic regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (C, max_iter, etc.)

    Returns:
        Trained logistic regression model
    """
    # TODO: Implement logistic regression training
    pass


def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (n_estimators, max_depth, learning_rate, etc.)

    Returns:
        Trained XGBoost model
    """
    # TODO: Implement XGBoost training
    pass


def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (n_estimators, max_depth, min_samples_split, etc.)

    Returns:
        Trained Random Forest model
    """
    # TODO: Implement Random Forest training
    pass


def train_svm(X_train, y_train, params=None):
    """
    Train a Support Vector Machine model.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (C, kernel, gamma, etc.)

    Returns:
        Trained SVM model
    """
    # TODO: Implement SVM training
    pass


def save_model(model, name):
    """
    Save model to disk using joblib.

    Args:
        model: Trained model to save
        name: Model name (e.g., 'logistic', 'xgboost')

    Returns:
        Path to saved model file
    """
    # TODO: Implement model saving
    # Hint: Create path using config.FEATURES_DIR.parent / "models" / f"{name}.joblib"
    #       Use joblib.dump(model, path)
    pass
