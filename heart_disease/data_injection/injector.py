import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORRUPTED_DIR = PROCESSED_DIR / "corrupted"


def set_seed(seed=42):
    np.random.seed(seed)


def inject_missing_data(df, ratio=0.1, columns=None, seed=42):
    df = df.copy()
    set_seed(seed)
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        n = int(len(df) * ratio)
        idx = np.random.choice(df.index, size=n, replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_gaussian_noise(df, columns=None, noise_level=0.05, seed=42):
    df = df.copy()
    set_seed(seed)
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            noise = np.random.normal(0, noise_level * df[col].std(), len(df))
            df[col] = df[col] + noise
            df[col] = df[col].clip(lower=0)
    return df


def inject_outliers(df, target_cols=None, spike_factor=3.0, days=7, seed=42):
    df = df.copy()
    set_seed(seed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_cols is not None:
        cols_to_spike = [target_cols]
    else:
        cols_to_spike = numeric_cols[:2]
    start = np.random.randint(0, max(1, len(df) - days))
    for col in cols_to_spike:
        df.loc[start:start + days - 1, col] = df.loc[start:start + days - 1, col] * spike_factor
    return df



def inject_systematic_bias(df, bias_factor=0.7, columns=None, seed=42):
    df = df.copy()
    set_seed(seed)
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col] * bias_factor
    return df


def inject_schema_drift(df, old_col=None, new_col=None):
    df = df.copy()
    if old_col and new_col and old_col in df.columns:
        df = df.rename(columns={old_col: new_col})
    return df


def generate_corrupted_dataset(df, preset="missing_light", target_col=None, seed=42):
    presets = {
        "missing_light": lambda d: inject_missing_data(d, ratio=0.05, columns=[target_col] if target_col else None, seed=seed),
        "missing_heavy": lambda d: inject_missing_data(d, ratio=0.20, columns=[target_col] if target_col else None, seed=seed),
        "noise_low": lambda d: inject_gaussian_noise(d, columns=[target_col] if target_col else None, noise_level=0.05, seed=seed),
        "noise_high": lambda d: inject_gaussian_noise(d, columns=[target_col] if target_col else None, noise_level=0.15, seed=seed),
        "outliers": lambda d: inject_outliers(d, target_col=target_col, spike_factor=3.0, days=5, seed=seed),
        "bias": lambda d: inject_systematic_bias(d, bias_factor=0.7, columns=[target_col] if target_col else None, seed=seed),
        "schema_drift": lambda d: inject_schema_drift(d, old_col=target_col, new_col=f"target_{target_col}" if target_col else None),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    return presets[preset](df)


if __name__ == "__main__":
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED_DIR / "clean_data.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        for preset in ["missing_light", "noise_high", "outliers", "bias"]:
            corrupted = generate_corrupted_dataset(df, preset=preset, seed=42)
            out_path = CORRUPTED_DIR / f"corrupted_{preset}.csv"
            corrupted.to_csv(out_path, index=False)
            print(f"Generated: {out_path}")
