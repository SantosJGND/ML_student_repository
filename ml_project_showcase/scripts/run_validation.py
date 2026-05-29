#!/usr/bin/env python3
"""
Data Validation Script
Validates input data (clean or corrupted) using Pydantic and Pandera.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger
from data_validation import validate_with_pydantic, validate_with_pandera

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate breast cancer data")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to CSV file to validate (clean or corrupted)")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    valid_df, invalid_df = validate_with_pydantic(df)
    print(f"\nPydantic validation: {len(valid_df)} valid, {len(invalid_df)} invalid")

    if len(invalid_df) > 0:
        print("\nSample invalid rows:")
        print(invalid_df[["_validation_error"]].head(5).to_string())

    success, errors = validate_with_pandera(df)
    print(f"\nPandera validation: {'PASSED' if success else 'FAILED'}")
    if errors:
        print(f"Errors:\n{errors[:2000]}")

    all_valid = len(invalid_df) == 0 and success
    logger.info(f"Validation completed: {'ALL PASS' if all_valid else 'FAILURES DETECTED'}")
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
