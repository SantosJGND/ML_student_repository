import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from injector import generate_corrupted_dataset
import pandas as pd
from config import CORRUPTION_PRESETS, RANDOM_SEED, TARGET_COL

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_TRAIN = PROJECT_ROOT / "data" / "processed" / "clean_train.csv"
CLEAN_TEST = PROJECT_ROOT / "data" / "processed" / "clean_test.csv"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "processed" / "corrupted"


def main():
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)

    for name, path in [("train", CLEAN_TRAIN), ("test", CLEAN_TEST)]:
        df = pd.read_csv(path)
        print(f"Loaded clean {name}: {df.shape}")

        for preset_name in CORRUPTION_PRESETS:
            corrupted = generate_corrupted_dataset(
                df, preset=preset_name, target_col=TARGET_COL, seed=RANDOM_SEED
            )
            out_path = CORRUPTED_DIR / f"corrupted_{name}_{preset_name}.csv"
            corrupted.to_csv(out_path, index=False)
            print(f"  Generated: {out_path} ({corrupted.shape})")


if __name__ == "__main__":
    main()
