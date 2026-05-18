import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from injector import generate_corrupted_dataset
import pandas as pd
from config import CORRUPTION_PRESETS, RANDOM_SEED, TARGET_COL

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "processed" / "corrupted"


def main():
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATA)
    print(f"Loaded clean data: {df.shape}")

    for preset_name, params in CORRUPTION_PRESETS.items():
        corrupted = generate_corrupted_dataset(df, params, seed=RANDOM_SEED)
        out_path = CORRUPTED_DIR / f"corrupted_{preset_name}.csv"
        corrupted.to_csv(out_path, index=False)
        print(f"  Generated: {out_path} ({corrupted.shape})")


if __name__ == "__main__":
    main()
