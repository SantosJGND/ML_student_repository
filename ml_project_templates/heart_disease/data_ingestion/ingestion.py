import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "processed.cleveland.data", header=None, names=column_names)

    df = df.replace("?", pd.NA).dropna().reset_index(drop=True)
    df["ca"] = df["ca"].astype(float)
    df["thal"] = df["thal"].astype(float)

    df["target"] = (df["num"] > 0).astype(int)
    clean = df.drop(columns=["num"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "clean_data.csv", index=False)
    return clean


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Clean data shape: {df.shape}")
    print(df.head())
