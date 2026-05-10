import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RELATIONAL_DIR = RAW_DIR / "relational"

RANDOM_SEED = 42


def generate_relational_data():
    RELATIONAL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / "SeoulBikeData.csv", encoding="unicode_escape")
    n = len(df)

    rng = np.random.default_rng(RANDOM_SEED)
    df["rental_id"] = range(1, n + 1)

    weather_cols = [
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
        "Visibility (10m)", "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
    ]

    # ---- Table 1: bike_rentals (all 8760 rows) ----
    rentals = df[["rental_id", "Date", "Hour", "Rented Bike Count"]].copy()
    rentals.to_csv(RELATIONAL_DIR / "bike_rentals.csv", index=False)
    print(f"bike_rentals: {len(rentals)} rows")

    # ---- Table 2: weather_conditions ----
    weather = df[["rental_id", "Date"] + weather_cols].copy()
    drop_weather_idx = rng.choice(weather.index, size=60, replace=False)
    weather = weather.drop(drop_weather_idx)
    weather = weather.rename(columns={"rental_id": "weather_id"})

    fake_weather_ids = range(n + 1, n + 31)
    real_weather = weather[weather_cols]
    sample_rows = real_weather.sample(n=30, replace=True, random_state=rng)
    noise = rng.normal(0, 0.05, size=sample_rows.shape)
    perturbed = sample_rows * (1 + noise)
    perturbed["Rainfall(mm)"] = perturbed["Rainfall(mm)"].clip(lower=0)
    perturbed["Snowfall (cm)"] = perturbed["Snowfall (cm)"].clip(lower=0)

    fake_dates = rng.choice(df["Date"].unique(), size=30).tolist()

    fake_weather = pd.DataFrame({
        "weather_id": list(fake_weather_ids),
        "Date": fake_dates,
    })
    for col in weather_cols:
        fake_weather[col] = perturbed[col].values

    weather = pd.concat([weather, fake_weather], ignore_index=True)
    weather.to_csv(RELATIONAL_DIR / "weather_conditions.csv", index=False)
    print(f"weather_conditions: {len(weather)} rows (incl. 30 fake)")

    # ---- Table 3: time_context ----
    context = df[["rental_id", "Date", "Seasons", "Holiday", "Functioning Day"]].copy()
    drop_context_idx = rng.choice(context.index, size=30, replace=False)
    context = context.drop(drop_context_idx)
    context = context.rename(columns={"rental_id": "context_id"})

    fake_context_ids = range(n + 31, n + 51)
    sample_ctx = context[["Seasons", "Holiday", "Functioning Day"]].sample(n=20, replace=True, random_state=rng)
    fake_context = pd.DataFrame({
        "context_id": list(fake_context_ids),
        "Date": rng.choice(df["Date"].unique(), size=20).tolist(),
    })
    for col in ["Seasons", "Holiday", "Functioning Day"]:
        fake_context[col] = sample_ctx[col].values

    context = pd.concat([context, fake_context], ignore_index=True)
    context.to_csv(RELATIONAL_DIR / "time_context.csv", index=False)
    print(f"time_context: {len(context)} rows (incl. 20 fake)")

    print("\nDone. Files saved to:", RELATIONAL_DIR)


def verify_merge_integrity():
    original = pd.read_csv(RAW_DIR / "SeoulBikeData.csv", encoding="unicode_escape")
    original["rental_id"] = range(1, len(original) + 1)

    rentals = pd.read_csv(RELATIONAL_DIR / "bike_rentals.csv")
    weather = pd.read_csv(RELATIONAL_DIR / "weather_conditions.csv")
    context = pd.read_csv(RELATIONAL_DIR / "time_context.csv")

    weather_cols = [
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
        "Visibility (10m)", "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
    ]
    context_cols = ["Seasons", "Holiday", "Functioning Day"]

    assert len(rentals) == 8760, f"Expected 8760 rentals, got {len(rentals)}"

    merged = rentals.merge(weather, left_on="rental_id", right_on="weather_id", how="left")
    merged = merged.merge(context, left_on="rental_id", right_on="context_id", how="left")

    assert len(merged) == 8760, (
        f"LEFT join should preserve all 8760 rentals, got {len(merged)}"
    )
    assert merged[weather_cols].isna().any(axis=1).sum() == 60, (
        f"Expected exactly 60 rows with missing weather"
    )
    assert merged.context_id.isna().sum() == 30, (
        f"Expected exactly 30 rows with missing context"
    )

    inner = rentals.merge(weather, left_on="rental_id", right_on="weather_id", how="inner")
    inner = inner.merge(context, left_on="rental_id", right_on="context_id", how="inner")
    assert len(inner) == 8670, (
        f"INNER join of all 3 should give 8670 rows, got {len(inner)}"
    )

    overlap = merged.dropna(subset=weather_cols + context_cols).copy()
    overlap = overlap.merge(
        original, on="rental_id", how="inner", suffixes=("_merged", "_orig")
    )

    for col in weather_cols:
        merged_vals = overlap[f"{col}_merged"]
        orig_vals = overlap[f"{col}_orig"]
        assert (merged_vals == orig_vals).all(), (
            f"Mismatch in {col}: merged != original for matching rows"
        )

    for col in context_cols:
        merged_vals = overlap[f"{col}_merged"]
        orig_vals = overlap[f"{col}_orig"]
        assert (merged_vals == orig_vals).all(), (
            f"Mismatch in {col}: merged != original for matching rows"
        )

    assert (overlap["Hour_merged"] == overlap["Hour_orig"]).all()
    assert (overlap["Rented Bike Count_merged"] == overlap["Rented Bike Count_orig"]).all()

    print("✅ Merge integrity verified: overlapping rows match the original data exactly")
    print(f"   LEFT join preserves {len(merged)} rows (60 missing weather, 30 missing context)")
    print(f"   INNER join gives {len(inner)} rows (8670 = 8760 - 60 - 30)")
    print(f"   {len(overlap)} rows have complete data and match the original")


if __name__ == "__main__":
    generate_relational_data()
    verify_merge_integrity()
