import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RELATIONAL_DIR = RAW_DIR / "relational"

RANDOM_SEED = 42

attributes_cont = [
    "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "ConcavePoints", "Symmetry", "FractalDimension",
]
attributes_expanded = [
    f"{stat}_{feat}" for feat in attributes_cont for stat in ["Mean", "SE", "Worst"]
]
mean_cols = [c for c in attributes_expanded if c.startswith("Mean_")]
se_cols = [c for c in attributes_expanded if c.startswith("SE_")]
worst_cols = [c for c in attributes_expanded if c.startswith("Worst_")]


def generate_relational_data():
    RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)

    col_names = ["ID", "Diagnosis"] + mean_cols + se_cols + worst_cols
    df = pd.read_csv(RAW_DIR / "wdbc.data", header=None, names=col_names)
    n = len(df)

    rng = np.random.default_rng(RANDOM_SEED)
    df["sample_id"] = range(1, n + 1)

    # ---- Table 1: patient_diagnosis (all 569 rows) ----
    diagnosis = df[["sample_id", "Diagnosis"]].copy()
    diagnosis.to_csv(RELATIONAL_DIR / "patient_diagnosis.csv", index=False)
    print(f"patient_diagnosis: {len(diagnosis)} rows")

    # ---- Table 2: mean_measurements (Mean + SE, 20 cols) ----
    mean_tbl = df[["sample_id"] + mean_cols + se_cols].copy()
    drop_mean_idx = rng.choice(mean_tbl.index, size=15, replace=False)
    mean_tbl = mean_tbl.drop(drop_mean_idx)
    mean_tbl = mean_tbl.rename(columns={"sample_id": "mean_id"})

    fake_mean_ids = range(n + 1, n + 9)
    real_mean = mean_tbl[mean_cols + se_cols]
    sample_rows = real_mean.sample(n=8, replace=True, random_state=rng)
    noise = rng.normal(0, 0.02, size=sample_rows.shape)
    perturbed = sample_rows * (1 + noise)
    perturbed = perturbed.clip(lower=0)

    fake_mean = pd.DataFrame({"mean_id": list(fake_mean_ids)})
    for col in mean_cols + se_cols:
        fake_mean[col] = perturbed[col].values

    mean_tbl = pd.concat([mean_tbl, fake_mean], ignore_index=True)
    mean_tbl.to_csv(RELATIONAL_DIR / "mean_measurements.csv", index=False)
    print(f"mean_measurements: {len(mean_tbl)} rows (incl. 8 fake)")

    # ---- Table 3: worst_measurements (10 Worst cols) ----
    worst_tbl = df[["sample_id"] + worst_cols].copy()
    drop_worst_idx = rng.choice(worst_tbl.index, size=20, replace=False)
    worst_tbl = worst_tbl.drop(drop_worst_idx)
    worst_tbl = worst_tbl.rename(columns={"sample_id": "worst_id"})

    fake_worst_ids = range(n + 9, n + 15)
    real_worst = worst_tbl[worst_cols]
    sample_rows_w = real_worst.sample(n=6, replace=True, random_state=rng)
    noise_w = rng.normal(0, 0.02, size=sample_rows_w.shape)
    perturbed_w = sample_rows_w * (1 + noise_w)
    perturbed_w = perturbed_w.clip(lower=0)

    fake_worst = pd.DataFrame({"worst_id": list(fake_worst_ids)})
    for col in worst_cols:
        fake_worst[col] = perturbed_w[col].values

    worst_tbl = pd.concat([worst_tbl, fake_worst], ignore_index=True)
    worst_tbl.to_csv(RELATIONAL_DIR / "worst_measurements.csv", index=False)
    print(f"worst_measurements: {len(worst_tbl)} rows (incl. 6 fake)")

    print("\nDone. Files saved to:", RELATIONAL_DIR)


def verify_merge_integrity():
    col_names = ["ID", "Diagnosis"] + mean_cols + se_cols + worst_cols
    original = pd.read_csv(RAW_DIR / "wdbc.data", header=None, names=col_names)
    original["sample_id"] = range(1, len(original) + 1)

    diagnosis = pd.read_csv(RELATIONAL_DIR / "patient_diagnosis.csv")
    mean_tbl = pd.read_csv(RELATIONAL_DIR / "mean_measurements.csv")
    worst_tbl = pd.read_csv(RELATIONAL_DIR / "worst_measurements.csv")

    assert len(diagnosis) == 569, f"Expected 569 diagnosis rows, got {len(diagnosis)}"

    merged = diagnosis.merge(mean_tbl, left_on="sample_id", right_on="mean_id", how="left")
    merged = merged.merge(worst_tbl, left_on="sample_id", right_on="worst_id", how="left")

    assert len(merged) == 569, (
        f"LEFT join should preserve all 569 rows, got {len(merged)}"
    )
    nan_mean = merged[mean_cols + se_cols].isna().any(axis=1).sum()
    nan_worst = merged[worst_cols].isna().any(axis=1).sum()
    print(f"   Missing mean: {nan_mean}, Missing worst: {nan_worst}")

    overlap_both = merged[mean_cols + se_cols].isna().any(axis=1) & merged[worst_cols].isna().any(axis=1)
    overlap_both = overlap_both.sum()

    inner = diagnosis.merge(mean_tbl, left_on="sample_id", right_on="mean_id", how="inner")
    inner = inner.merge(worst_tbl, left_on="sample_id", right_on="worst_id", how="inner")
    expected_inner = 569 - nan_mean - nan_worst + overlap_both
    assert len(inner) == expected_inner, (
        f"INNER join should give {expected_inner} rows, got {len(inner)}"
    )

    overlap = merged.dropna(subset=mean_cols + se_cols + worst_cols).copy()
    overlap = overlap.merge(original, on="sample_id", how="inner", suffixes=("_merged", "_orig"))

    for col in mean_cols + se_cols + worst_cols:
        merged_vals = overlap[f"{col}_merged"]
        orig_vals = overlap[f"{col}_orig"]
        assert (np.abs(merged_vals - orig_vals) < 1e-10).all(), (
            f"Mismatch in {col}: merged != original for matching rows"
        )

    print("✅ Merge integrity verified: overlapping rows match the original exactly")
    print(f"   LEFT join: {len(merged)} rows ({nan_mean} missing mean, {nan_worst} missing worst)")
    print(f"   INNER join: {len(inner)} rows ({569} - {nan_mean} - {nan_worst} + {overlap_both} overlap)")
    print(f"   {len(overlap)} complete rows match original")


if __name__ == "__main__":
    generate_relational_data()
    verify_merge_integrity()
