import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def load_and_clean():
    train = pd.read_csv(RAW_DIR / "adult.data", header=None, names=column_names)
    test = pd.read_csv(RAW_DIR / "adult.test", header=None, names=column_names, skiprows=1)

    for df in [train, test]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
        df.replace("?", pd.NA, inplace=True)
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    train["income"] = train["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    test["income"] = test["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

    train_encoded = pd.get_dummies(train, drop_first=True)
    test_encoded = pd.get_dummies(test, drop_first=True)

    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_encoded.to_csv(PROCESSED_DIR / "clean_train.csv", index=False)
    test_encoded.to_csv(PROCESSED_DIR / "clean_test.csv", index=False)
    return train_encoded, test_encoded


if __name__ == "__main__":
    train_df, test_df = load_and_clean()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(train_df.head())
