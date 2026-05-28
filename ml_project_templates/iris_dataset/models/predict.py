"""
Model Prediction Module
Contains functions for model inference and prediction.
Multiclass classification (3 species).
"""


def predict(model, X_test):
    """
    Generate predictions using trained model.

    Args:
        model: Trained model object
        X_test: Test features

    Returns:
        Array of predictions (0: setosa, 1: versicolor, 2: virginica)
    """
    # TODO: Implement prediction logic
    pass


def predict_proba(model, X_test):
    """
    Generate probability predictions using trained model.
    Multiclass classification (3 species).

    Args:
        model: Trained model object
        X_test: Test features

    Returns:
        Array of class probabilities (n_samples x 3 classes)
    """
    # TODO: Implement probability prediction logic
    pass


def predict_species(model_name, features_dict):
    """
    Predict iris species (0=setosa, 1=versicolor, 2=virginica).

    Args:
        model_name: Name of model to use
        features_dict: Dict with 4 feature values

    Returns:
        Predicted species (0/1/2), or None if model fails
    """
    # TODO: Implement species prediction
    # Hint: Load model, call predict() with features_dict
    pass


def predict_species_proba(model_name, features_dict):
    """
    Predict probability distribution across species.

    Args:
        model_name: Name of model to use (should support predict_proba)
        features_dict: Dict with 4 feature values

    Returns:
        Dict with probabilities for each species, or None if fails
        e.g., {"setosa": 0.1, "versicolor": 0.3, "virginica": 0.6}
    """
    # TODO: Implement species probability prediction
    # Hint: Load model, call predict_proba(), convert to dict
    # probs = model.predict_proba(X)
    # return {"setosa": float(probs[0][0]), ...}
    pass
