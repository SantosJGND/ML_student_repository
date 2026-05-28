import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RELATIONAL_DIR = RAW_DIR / "relational"

RANDOM_SEED = 42

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

clinical_cols = [
    "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "processed.cleveland.data", header=None, names=column_names)
    df = df.replace("?", pd.NA).dropna().reset_index(drop=True)
    df["ca"] = df["ca"].astype(float)
    df["thal"] = df["thal"].astype(float)
    df["target"] = (df["num"] > 0).astype(int)
    return df


def generate_relational_data():
    RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    n = len(df)

    rng = np.random.default_rng(RANDOM_SEED)
    df["sample_id"] = range(1, n + 1)

    # ---- Table 1: patient (demographics, all rows) ----
    patient = df[["sample_id", "age", "sex"]].copy()
    patient.to_csv(RELATIONAL_DIR / "patient.csv", index=False)
    print(f"patient: {len(patient)} rows")

    # ---- Table 2: clinical (test results) ----
    clinical = df[["sample_id"] + clinical_cols].copy()
    drop_clinical_idx = rng.choice(clinical.index, size=8, replace=False)
    clinical = clinical.drop(drop_clinical_idx)
    clinical = clinical.rename(columns={"sample_id": "clinical_id"})

    fake_clinical_ids = range(n + 1, n + 6)
    real_clinical = clinical[clinical_cols]
    sample_rows = real_clinical.sample(n=5, replace=True, random_state=rng)
    noise_cat = rng.integers(0, 2, size=(5, 5))
    noise_cont = rng.normal(1, 0.03, size=(5, 6))

    fake_clinical = pd.DataFrame({"clinical_id": list(fake_clinical_ids)})
    for i, col in enumerate(clinical_cols[:5]):
        vals = sample_rows[col].values.copy()
        flip = noise_cat[:, i] == 1
        vals[flip] = rng.choice(real_clinical[col].unique(), size=flip.sum())
        fake_clinical[col] = vals
    for i, col in enumerate(clinical_cols[5:]):
        fake_clinical[col] = np.clip(sample_rows[col].values * noise_cont[:, i], 0, None)

    clinical = pd.concat([clinical, fake_clinical], ignore_index=True)
    clinical.to_csv(RELATIONAL_DIR / "clinical.csv", index=False)
    print(f"clinical: {len(clinical)} rows (incl. 5 fake)")

    # ---- Table 3: outcome (diagnosis target) ----
    outcome = df[["sample_id", "target"]].copy()
    drop_outcome_idx = rng.choice(outcome.index, size=12, replace=False)
    outcome = outcome.drop(drop_outcome_idx)
    outcome = outcome.rename(columns={"sample_id": "outcome_id"})

    fake_outcome_ids = range(n + 6, n + 13)
    fake_outcome = pd.DataFrame({
        "outcome_id": list(fake_outcome_ids),
        "target": rng.integers(0, 2, size=7).tolist(),
    })

    outcome = pd.concat([outcome, fake_outcome], ignore_index=True)
    outcome.to_csv(RELATIONAL_DIR / "outcome.csv", index=False)
    print(f"outcome: {len(outcome)} rows (incl. 7 fake)")

    print("\nDone. Files saved to:", RELATIONAL_DIR)


def verify_merge_integrity():
    original = load_and_clean()
    original["sample_id"] = range(1, len(original) + 1)

    patient = pd.read_csv(RELATIONAL_DIR / "patient.csv")
    clinical = pd.read_csv(RELATIONAL_DIR / "clinical.csv")
    outcome = pd.read_csv(RELATIONAL_DIR / "outcome.csv")

    assert len(patient) == len(original), (
        f"Expected {len(original)} patient rows, got {len(patient)}"
    )

    merged = patient.merge(clinical, left_on="sample_id", right_on="clinical_id", how="left")
    merged = merged.merge(outcome, left_on="sample_id", right_on="outcome_id", how="left")

    assert len(merged) == len(original), (
        f"LEFT join should preserve all {len(original)} rows, got {len(merged)}"
    )

    nan_clinical = merged[clinical_cols].isna().any(axis=1).sum()
    nan_outcome = merged["target"].isna().sum()
    print(f"   Missing clinical: {nan_clinical}, Missing outcome: {nan_outcome}")

    overlap_both = (merged[clinical_cols].isna().any(axis=1) & merged["target"].isna()).sum()

    inner = patient.merge(clinical, left_on="sample_id", right_on="clinical_id", how="inner")
    inner = inner.merge(outcome, left_on="sample_id", right_on="outcome_id", how="inner")
    expected_inner = len(original) - nan_clinical - nan_outcome + overlap_both
    assert len(inner) == expected_inner, (
        f"INNER join should give {expected_inner} rows, got {len(inner)}"
    )

    overlap = merged.dropna(subset=clinical_cols + ["target"]).copy()
    if len(overlap) > 0:
        overlap = overlap.merge(original, on="sample_id", how="inner", suffixes=("_merged", "_orig"))

        for col in clinical_cols:
            mv = overlap[f"{col}_merged"]
            ov = overlap[f"{col}_orig"]
            assert (np.abs(mv - ov) < 1e-10).all(), f"Mismatch in {col}"

        assert (overlap["target_merged"] == overlap["target_orig"]).all()

    print("✅ Merge integrity verified: overlapping rows match the original exactly")
    print(f"   LEFT join: {len(merged)} rows ({nan_clinical} missing clinical, {nan_outcome} missing outcome)")
    print(f"   INNER join: {len(inner)} rows")
    print(f"   {len(overlap)} complete rows match original")


if __name__ == "__main__":
    generate_relational_data()
    verify_merge_integrity()
