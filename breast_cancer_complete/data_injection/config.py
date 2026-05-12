import numpy as np

RANDOM_SEED = 42

TARGET_COL = "Diagnosis"

NUMERIC_COLS = [
    "Mean_Radius", "Mean_Texture", "Mean_Perimeter", "Mean_Area", "Mean_Smoothness",
    "Mean_Compactness", "Mean_Concavity", "Mean_ConcavePoints", "Mean_Symmetry", "Mean_FractalDimension",
    "SE_Radius", "SE_Texture", "SE_Perimeter", "SE_Area", "SE_Smoothness",
    "SE_Compactness", "SE_Concavity", "SE_ConcavePoints", "SE_Symmetry", "SE_FractalDimension",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness",
    "Worst_Compactness", "Worst_Concavity", "Worst_ConcavePoints", "Worst_Symmetry", "Worst_FractalDimension",
]

CORRUPTION_PRESETS = {
    "missing_light": {"type": "missing", "ratio": 0.05, "columns": [TARGET_COL]},
    "missing_heavy": {"type": "missing", "ratio": 0.20, "columns": [TARGET_COL]},
    "noise_low": {"type": "noise", "noise_level": 0.05, "columns": NUMERIC_COLS[:5]},
    "noise_high": {"type": "noise", "noise_level": 0.35, "columns": NUMERIC_COLS[:5]},
    "outliers": {"type": "outliers", "spike_factor": 16.0, "days": 25, "target_col": NUMERIC_COLS[:10]},
    "bias": {"type": "bias", "bias_factor": 0.9, "columns": NUMERIC_COLS[:5]},
    "schema_drift": {"type": "schema_drift", "old_col": TARGET_COL, "new_col": "diagnosis_label"},
}
