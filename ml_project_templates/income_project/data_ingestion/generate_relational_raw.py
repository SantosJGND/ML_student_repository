import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RELATIONAL_DIR = RAW_DIR / "relational"

RANDOM_SEED = 42

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

personal_cols = ["age", "sex", "race", "native-country"]

employment_cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship"]
employment_num_cols = ["education-num", "hours-per-week"]
employment_cols = employment_cat_cols + employment_num_cols

census_cols = ["fnlwgt", "capital-gain", "capital-loss", "income"]


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "adult.data", header=None, names=column_names)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    df = df.replace("?", pd.NA).dropna().reset_index(drop=True)
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    return df


def generate_relational_data():
    RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    n = len(df)

    rng = np.random.default_rng(RANDOM_SEED)
    df["sample_id"] = range(1, n + 1)

    # ---- Table 1: personal (demographics, all rows) ----
    personal = df[["sample_id"] + personal_cols].copy()
    personal.to_csv(RELATIONAL_DIR / "personal.csv", index=False)
    print(f"personal: {len(personal)} rows")

    # ---- Table 2: employment (work/education) ----
    employment = df[["sample_id"] + employment_cols].copy()
    drop_emp_idx = rng.choice(employment.index, size=200, replace=False)
    employment = employment.drop(drop_emp_idx)
    employment = employment.rename(columns={"sample_id": "employment_id"})

    fake_emp_ids = range(n + 1, n + 41)
    real_emp = employment[employment_cols]

    fake_emp = pd.DataFrame({"employment_id": list(fake_emp_ids)})

    sample_cat = real_emp[employment_cat_cols].sample(n=40, replace=True, random_state=rng)
    for col in employment_cat_cols:
        flip = rng.choice([True, False], size=40, p=[0, 1])
        vals = sample_cat[col].values.copy()
        vals[flip] = rng.choice(real_emp[col].unique(), size=flip.sum())
        fake_emp[col] = vals

    sample_num = real_emp[employment_num_cols].sample(n=40, replace=True, random_state=rng)
    for col in employment_num_cols:
        fake_emp[col] = np.clip(
            (sample_num[col].values * rng.normal(1, 0.03, size=40)),
            0, None
        ).round().astype(int)

    employment = pd.concat([employment, fake_emp], ignore_index=True)
    employment.to_csv(RELATIONAL_DIR / "employment.csv", index=False)
    print(f"employment: {len(employment)} rows (incl. 40 fake)")

    # ---- Table 3: census (financial + target) ----
    census = df[["sample_id"] + census_cols].copy()
    drop_cen_idx = rng.choice(census.index, size=300, replace=False)
    census = census.drop(drop_cen_idx)
    census = census.rename(columns={"sample_id": "census_id"})

    fake_cen_ids = range(n + 41, n + 91)
    real_cen = census[census_cols]

    fake_cen = pd.DataFrame({"census_id": list(fake_cen_ids)})

    sample_cen = real_cen.sample(n=50, replace=True, random_state=rng)
    fake_cen["fnlwgt"] = np.clip(
        (sample_cen["fnlwgt"].values * rng.normal(1, 0.05, size=50)),
        0, None
    ).round().astype(int)
    fake_cen["capital-gain"] = np.clip(
        (sample_cen["capital-gain"].values * rng.normal(1, 0.1, size=50)),
        0, None
    ).round().astype(int)
    fake_cen["capital-loss"] = np.clip(
        (sample_cen["capital-loss"].values * rng.normal(1, 0.1, size=50)),
        0, None
    ).round().astype(int)
    fake_cen["income"] = rng.integers(0, 2, size=50)

    census = pd.concat([census, fake_cen], ignore_index=True)
    census.to_csv(RELATIONAL_DIR / "census.csv", index=False)
    print(f"census: {len(census)} rows (incl. 50 fake)")

    print("\nDone. Files saved to:", RELATIONAL_DIR)


def verify_merge_integrity():
    original = load_and_clean()
    original["sample_id"] = range(1, len(original) + 1)

    personal = pd.read_csv(RELATIONAL_DIR / "personal.csv")
    employment = pd.read_csv(RELATIONAL_DIR / "employment.csv")
    census = pd.read_csv(RELATIONAL_DIR / "census.csv")

    assert len(personal) == len(original), (
        f"Expected {len(original)} personal rows, got {len(personal)}"
    )

    merged = personal.merge(employment, left_on="sample_id", right_on="employment_id", how="left")
    merged = merged.merge(census, left_on="sample_id", right_on="census_id", how="left")

    assert len(merged) == len(original), (
        f"LEFT join should preserve all {len(original)} rows, got {len(merged)}"
    )

    nan_emp = merged[employment_cols].isna().any(axis=1).sum()
    nan_cen = merged[census_cols].isna().any(axis=1).sum()
    print(f"   Missing employment: {nan_emp}, Missing census: {nan_cen}")

    overlap_both = (merged[employment_cols].isna().any(axis=1) & merged[census_cols].isna().any(axis=1)).sum()

    inner = personal.merge(employment, left_on="sample_id", right_on="employment_id", how="inner")
    inner = inner.merge(census, left_on="sample_id", right_on="census_id", how="inner")
    expected_inner = len(original) - nan_emp - nan_cen + overlap_both
    assert len(inner) == expected_inner, (
        f"INNER join should give {expected_inner} rows, got {len(inner)}"
    )

    overlap = merged.dropna(subset=employment_cols + census_cols).copy()
    if len(overlap) > 0:
        overlap = overlap.merge(original, on="sample_id", how="inner", suffixes=("_merged", "_orig"))

        for col in employment_num_cols + ["fnlwgt", "capital-gain", "capital-loss"]:
            mv = overlap[f"{col}_merged"]
            ov = overlap[f"{col}_orig"]
            assert (mv == ov).all(), f"Mismatch in numeric {col}"

        for col in employment_cat_cols + ["income"]:
            mv_ser = overlap[f"{col}_merged"]
            ov_ser = overlap[f"{col}_orig"]
            assert (mv_ser == ov_ser).all(), f"Mismatch in col {col}"

        assert (overlap["income_merged"] == overlap["income_orig"]).all()

    print("✅ Merge integrity verified: overlapping rows match the original exactly")
    print(f"   LEFT join: {len(merged)} rows ({nan_emp} missing employment, {nan_cen} missing census)")
    print(f"   INNER join: {len(inner)} rows")
    print(f"   {len(overlap)} complete rows match original")


if __name__ == "__main__":
    generate_relational_data()
    verify_merge_integrity()
