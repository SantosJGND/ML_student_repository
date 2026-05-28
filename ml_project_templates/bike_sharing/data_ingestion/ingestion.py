import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "SeoulBikeData.csv", encoding="unicode_escape")

    X = df.drop(columns=["Date", "Rented Bike Count"])
    y = df["Rented Bike Count"]

    X_encoded = pd.get_dummies(X, columns=["Seasons", "Holiday", "Functioning Day"], drop_first=True)

    clean = X_encoded.copy()
    clean["Rented Bike Count"] = y.values

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "clean_data.csv", index=False)
    return clean


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Clean data shape: {df.shape}")
    print(df.head())
