import numpy as np

RANDOM_SEED = 42

TARGET_COL = "Rented Bike Count"

NUMERIC_COLS = [
    "Hour", "Temperature(?C)", "Humidity(%)", "Wind speed (m/s)",
    "Visibility (10m)", "Dew point temperature(?C)", "Solar Radiation (MJ/m2)",
    "Rainfall(mm)", "Snowfall (cm)",
]

CATEGORICAL_COLS = ["Seasons", "Holiday", "Functioning Day"]

CORRUPTION_PRESETS = {
    "missing_light": {"type": "missing", "ratio": 0.05, "columns": [TARGET_COL]},
    "missing_heavy": {"type": "missing", "ratio": 0.20, "columns": [TARGET_COL]},
    "noise_low": {"type": "noise", "noise_level": 0.05, "columns": [TARGET_COL]},
    "noise_high": {"type": "noise", "noise_level": 0.15, "columns": [TARGET_COL]},
    "outliers": {"type": "outliers", "spike_factor": 3.0, "days": 5, "target_cols": NUMERIC_COLS[:5] + [TARGET_COL]},
    "bias": {"type": "bias", "bias_factor": 0.7, "columns": [TARGET_COL]},
    "schema_drift": {"type": "schema_drift", "old_col": TARGET_COL, "new_col": "bike_count"},
    "duplicates_light": {"type": "duplicates", "ratio": 0.05},
    "duplicates_heavy": {"type": "duplicates", "ratio": 0.25},
}
