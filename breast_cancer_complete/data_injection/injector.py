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


def generate_corrupted_dataset(df, params, seed=42):
    corr_type = params["type"]
    kwargs = {k: v for k, v in params.items() if k not in ("type", "seed")}
    seed = params.get("seed", seed)

    if corr_type == "missing":
        return inject_missing_data(df, seed=seed, **kwargs)
    elif corr_type == "noise":
        return inject_gaussian_noise(df, seed=seed, **kwargs)
    elif corr_type == "outliers":
        return inject_outliers(df, seed=seed, **kwargs)
    elif corr_type == "bias":
        return inject_systematic_bias(df, seed=seed, **kwargs)
    elif corr_type == "schema_drift":
        return inject_schema_drift(df, **kwargs)
    else:
        raise ValueError(f"Unknown corruption type: {corr_type}. Expected one of: missing, noise, outliers, bias, schema_drift")


if __name__ == "__main__":
    from config import CORRUPTION_PRESETS
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED_DIR / "clean_data.csv"
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        for name, params in CORRUPTION_PRESETS.items():
            corrupted = generate_corrupted_dataset(df, params, seed=42)
            out_path = CORRUPTED_DIR / f"corrupted_{name}.csv"
            corrupted.to_csv(out_path, index=False)
            print(f"Generated: {out_path}")
