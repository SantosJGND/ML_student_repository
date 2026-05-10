import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RELATIONAL_DIR = RAW_DIR / "relational"

RANDOM_SEED = 42

attribute_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
sepal_cols = ["sepal_length", "sepal_width"]
petal_cols = ["petal_length", "petal_width"]


def load_and_clean():
    df = pd.read_csv(RAW_DIR / "iris.data", header=None, names=attribute_names)
    df["class"] = df["class"].astype("category").cat.codes
    return df


def generate_relational_data():
    RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    n = len(df)

    rng = np.random.default_rng(RANDOM_SEED)
    df["sample_id"] = range(1, n + 1)

    # ---- Table 1: species (target, all 150 rows) ----
    species = df[["sample_id", "class"]].copy()
    species.to_csv(RELATIONAL_DIR / "species.csv", index=False)
    print(f"species: {len(species)} rows")

    # ---- Table 2: sepal (sepal measurements) ----
    sepal = df[["sample_id"] + sepal_cols].copy()
    drop_sepal_idx = rng.choice(sepal.index, size=5, replace=False)
    sepal = sepal.drop(drop_sepal_idx)
    sepal = sepal.rename(columns={"sample_id": "sepal_id"})

    fake_sepal_ids = range(n + 1, n + 4)
    real_sepal = sepal[sepal_cols]
    sample_rows = real_sepal.sample(n=3, replace=True, random_state=rng)
    noise = rng.normal(1, 0.03, size=sample_rows.shape)
    perturbed = np.clip(sample_rows * noise, 0, None)

    fake_sepal = pd.DataFrame({"sepal_id": list(fake_sepal_ids)})
    for i, col in enumerate(sepal_cols):
        fake_sepal[col] = perturbed[col].values

    sepal = pd.concat([sepal, fake_sepal], ignore_index=True)
    sepal.to_csv(RELATIONAL_DIR / "sepal.csv", index=False)
    print(f"sepal: {len(sepal)} rows (incl. 3 fake)")

    # ---- Table 3: petal (petal measurements) ----
    petal = df[["sample_id"] + petal_cols].copy()
    drop_petal_idx = rng.choice(petal.index, size=8, replace=False)
    petal = petal.drop(drop_petal_idx)
    petal = petal.rename(columns={"sample_id": "petal_id"})

    fake_petal_ids = range(n + 4, n + 9)
    real_petal = petal[petal_cols]
    sample_rows_p = real_petal.sample(n=5, replace=True, random_state=rng)
    noise_p = rng.normal(1, 0.03, size=sample_rows_p.shape)
    perturbed_p = np.clip(sample_rows_p * noise_p, 0, None)

    fake_petal = pd.DataFrame({"petal_id": list(fake_petal_ids)})
    for i, col in enumerate(petal_cols):
        fake_petal[col] = perturbed_p[col].values

    petal = pd.concat([petal, fake_petal], ignore_index=True)
    petal.to_csv(RELATIONAL_DIR / "petal.csv", index=False)
    print(f"petal: {len(petal)} rows (incl. 5 fake)")

    print("\nDone. Files saved to:", RELATIONAL_DIR)


def verify_merge_integrity():
    original = load_and_clean()
    original["sample_id"] = range(1, len(original) + 1)

    species = pd.read_csv(RELATIONAL_DIR / "species.csv")
    sepal = pd.read_csv(RELATIONAL_DIR / "sepal.csv")
    petal = pd.read_csv(RELATIONAL_DIR / "petal.csv")

    assert len(species) == 150, f"Expected 150 species rows, got {len(species)}"

    merged = species.merge(sepal, left_on="sample_id", right_on="sepal_id", how="left")
    merged = merged.merge(petal, left_on="sample_id", right_on="petal_id", how="left")

    assert len(merged) == 150, (
        f"LEFT join should preserve all 150 rows, got {len(merged)}"
    )

    nan_sepal = merged[sepal_cols].isna().any(axis=1).sum()
    nan_petal = merged[petal_cols].isna().any(axis=1).sum()
    print(f"   Missing sepal: {nan_sepal}, Missing petal: {nan_petal}")

    overlap_both = (merged[sepal_cols].isna().any(axis=1) & merged[petal_cols].isna().any(axis=1)).sum()

    inner = species.merge(sepal, left_on="sample_id", right_on="sepal_id", how="inner")
    inner = inner.merge(petal, left_on="sample_id", right_on="petal_id", how="inner")
    expected_inner = 150 - nan_sepal - nan_petal + overlap_both
    assert len(inner) == expected_inner, (
        f"INNER join should give {expected_inner} rows, got {len(inner)}"
    )

    overlap = merged.dropna(subset=sepal_cols + petal_cols).copy()
    if len(overlap) > 0:
        overlap = overlap.merge(original, on="sample_id", how="inner", suffixes=("_merged", "_orig"))

        for col in sepal_cols + petal_cols:
            mv = overlap[f"{col}_merged"]
            ov = overlap[f"{col}_orig"]
            assert (np.abs(mv - ov) < 1e-10).all(), f"Mismatch in {col}"

        assert (overlap["class_merged"] == overlap["class_orig"]).all()

    print("✅ Merge integrity verified: overlapping rows match the original exactly")
    print(f"   LEFT join: {len(merged)} rows ({nan_sepal} missing sepal, {nan_petal} missing petal)")
    print(f"   INNER join: {len(inner)} rows")
    print(f"   {len(overlap)} complete rows match original")


if __name__ == "__main__":
    generate_relational_data()
    verify_merge_integrity()
