import numpy as np

RANDOM_SEED = 42

TARGET_COL = "income"

NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]

CORRUPTION_PRESETS = {
    "missing_light": {"type": "missing", "ratio": 0.05, "columns": [TARGET_COL]},
    "missing_heavy": {"type": "missing", "ratio": 0.20, "columns": [TARGET_COL]},
    "noise_low": {"type": "noise", "noise_level": 0.05, "columns": NUMERIC_COLS},
    "noise_high": {"type": "noise", "noise_level": 0.15, "columns": NUMERIC_COLS},
    "outliers": {"type": "outliers", "spike_factor": 3.0, "days": 5, "target_col": TARGET_COL},
    "bias": {"type": "bias", "bias_factor": 0.7, "columns": NUMERIC_COLS},
    "schema_drift": {"type": "schema_drift", "old_col": TARGET_COL, "new_col": "salary_class"},
}
