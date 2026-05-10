import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

attributes_cont = [
    "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "ConcavePoints", "Symmetry", "FractalDimension",
]
attributes_expanded = [
    f"{stat}_{feat}" for feat in attributes_cont for stat in ["Mean", "SE", "Worst"]
]
col_names = ["ID", "Diagnosis"] + attributes_expanded


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "wdbc.data", header=None, names=col_names)

    df["Diagnosis"] = df["Diagnosis"].apply(lambda x: 1 if x == "M" else 0)

    clean = df.drop(columns=["ID"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "clean_data.csv", index=False)
    return clean


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Clean data shape: {df.shape}")
    print(df.head())
