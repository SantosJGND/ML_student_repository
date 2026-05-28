import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "iris.data", header=None, names=attribute_names)

    # Encode target: convert class labels to numeric codes (0, 1, 2)
    df["class_encoded"] = df["class"].astype('category').cat.codes

    clean = df.drop(columns=["class"])
    clean = clean.rename(columns={"class_encoded": "class"})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "clean_data.csv", index=False)
    return clean


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Clean data shape: {df.shape}")
    print(df.head())
