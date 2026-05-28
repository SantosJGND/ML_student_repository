"""
Model Training Module
Contains functions to train different ML models for iris classification.
Multiclass classification (3 species: setosa, versicolor, virginica).
"""


def train_logistic_regression(X_train, y_train, params=None):
    """
    Train a logistic regression model for multiclass classification.

    Args:
        X_train: Training features
        y_train: Training labels (0: setosa, 1: versicolor, 2: virginica)
        params: Dict of hyperparameters (C, max_iter, etc.)

    Returns:
        Trained logistic regression model (multiclass)
    """
    # TODO: Implement logistic regression training with multi_class='ovr'
    pass


def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost classifier for multiclass classification.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (n_estimators, max_depth, learning_rate, etc.)
                Note: Use eval_metric='mlogloss' for multiclass

    Returns:
        Trained XGBoost model (multiclass)
    """
    # TODO: Implement XGBoost training with objective='multi:softprob'
    pass


def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest classifier for multiclass classification.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (n_estimators, max_depth, min_samples_split, etc.)

    Returns:
        Trained Random Forest model (multiclass)
    """
    # TODO: Implement Random Forest training
    pass


def train_svm(X_train, y_train, params=None):
    """
    Train a Support Vector Machine model for multiclass classification.

    Args:
        X_train: Training features
        y_train: Training labels
        params: Dict of hyperparameters (C, kernel, gamma, etc.)
                Note: Use decision_function_shape='ovr' for multiclass

    Returns:
        Trained SVM model (multiclass)
    """
    # TODO: Implement SVM training with decision_function_shape='ovr'
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
