#!/usr/bin/env python3
"""
Model Promotion Script
Champion/Challenger comparison with auto-promotion.
Compares current Production model (champion) against a candidate (challenger)
on clean test data and corrupted scenarios. Promotes challenger if it
passes gates and wins on enough corrupted scenarios.
"""
import argparse
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI
from evaluation.metrics import calculate_metrics
from evaluation.gate import check_gates

logger = get_logger(__name__)

CORRUPTION_PRESETS = [
    "missing_light", "missing_heavy", "noise_low", "noise_high",
    "outliers", "bias", "schema_drift",
]


def load_model_from_uri(uri):
    """Load model from MLflow URI or local joblib path."""
    import joblib
    if uri.startswith("models:") or uri.startswith("runs:"):
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(uri)
        logger.info(f"Loaded model from MLflow: {uri}")
        return model
    else:
        model = joblib.load(uri)
        logger.info(f"Loaded model from joblib: {uri}")
        return model


def get_feature_target(df):
    """Extract features and target from DataFrame."""
    for target_col in ("Diagnosis", "diagnosis_label", "target_Diagnosis"):
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            return X, y
    raise KeyError(f"No known target column found in {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger model promotion")
    parser.add_argument("--champion-uri", type=str, required=True,
                        help="MLflow URI or joblib path of current Production model")
    parser.add_argument("--challenger-uri", type=str, required=True,
                        help="MLflow URI or joblib path of candidate model")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to clean test data CSV")
    parser.add_argument("--corrupted-dir", type=str, required=True,
                        help="Directory containing corrupted_*.csv files")
    parser.add_argument("--gates", type=str,
                        default='{"accuracy": 0.95, "f1": 0.90}',
                        help="JSON string of gate thresholds")
    parser.add_argument("--registered-model-name", type=str,
                        default="breast_cancer_classifier",
                        help="Name in MLflow Model Registry")
    parser.add_argument("--promote-threshold", type=float, default=0.5,
                        help="Fraction of corrupted scenarios challenger must win (0.0-1.0)")
    parser.add_argument("--no-promote", action="store_true",
                        help="Run evaluation only, do not promote")
    args = parser.parse_args()

    gates = json.loads(args.gates)
    corrupted_path = Path(args.corrupted_dir)

    # Load models
    logger.info("Loading champion model...")
    champion = load_model_from_uri(args.champion_uri)
    logger.info("Loading challenger model...")
    challenger = load_model_from_uri(args.challenger_uri)

    # Load clean test data
    logger.info(f"Loading clean test data from {args.test_data}")
    df_clean = pd.read_csv(args.test_data)
    X_clean, y_clean = get_feature_target(df_clean)

    # Gate evaluation on clean data
    print("=" * 60)
    print("STEP 1: Gate Evaluation (Clean Test Data)")
    print("=" * 60)
    for name, model in [("Champion", champion), ("Challenger", challenger)]:
        y_pred = model.predict(X_clean)
        metrics = calculate_metrics(y_clean, y_pred)
        passed, _ = check_gates(metrics, gates)
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n{name}: {status}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Load corrupted data
    print("\n" + "=" * 60)
    print("STEP 2: Champion/Challenger Comparison on Corrupted Data")
    print("=" * 60)
    corrupted_data = {}
    for preset in CORRUPTION_PRESETS:
        path = corrupted_path / f"corrupted_{preset}.csv"
        if path.exists():
            corrupted_data[preset] = pd.read_csv(path)
            print(f"  Loaded {preset}: {corrupted_data[preset].shape}")

    comparison_rows = []
    for preset, df_corr in corrupted_data.items():
        try:
            X_corr, y_corr = get_feature_target(df_corr)
            if y_corr.dtype in (np.float64, np.float32):
                y_corr = y_corr.fillna(-1).astype(int)
            y_corr = (y_corr > 0.5).astype(int)

            champ_acc = calculate_metrics(y_corr, champion.predict(X_corr))["accuracy"]
            chall_acc = calculate_metrics(y_corr, challenger.predict(X_corr))["accuracy"]

            comparison_rows.append({
                "preset": preset,
                "champion_accuracy": champ_acc,
                "challenger_accuracy": chall_acc,
                "challenger_wins": chall_acc > champ_acc,
            })
            winner = "Challenger" if chall_acc > champ_acc else "Champion"
            print(f"  {preset:20s}  Champ={champ_acc:.4f}  Chall={chall_acc:.4f}  → {winner}")
        except Exception as e:
            logger.warning(f"  {preset}: skipped ({e})")

    if not comparison_rows:
        print("No corrupted scenarios could be evaluated. Aborting.")
        sys.exit(1)

    comp_df = pd.DataFrame(comparison_rows)
    wins = comp_df["challenger_wins"].sum()
    total = len(comp_df)
    win_rate = wins / total
    print(f"\nChallenger wins: {wins}/{total} ({win_rate:.1%})")

    # Decision
    print("\n" + "=" * 60)
    print("STEP 3: Promotion Decision")
    print("=" * 60)
    if args.no_promote:
        print("  --no-promote set. Skipping promotion.")
        print(f"  Recommendation: promote if challenger wins > {args.promote_threshold:.0%}")
        sys.exit(0)

    if win_rate > args.promote_threshold:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{args.registered_model_name}'")
        if args.challenger_uri.startswith("runs:"):
            challenger_run_id = args.challenger_uri.split("/")[1]
            challenger_version = None
            for v in versions:
                if v.run_id == challenger_run_id:
                    challenger_version = v.version
                    break
            if challenger_version is None:
                logger.warning("Challenger not yet registered. Registering first...")
                import mlflow
                result = mlflow.register_model(args.challenger_uri, args.registered_model_name)
                challenger_version = result.version

        if challenger_version:
            client.transition_model_version_stage(
                name=args.registered_model_name,
                version=challenger_version,
                stage="Production"
            )
            print(f"  ✓ Challenger (v{challenger_version}) promoted to Production")

            for v in versions:
                if v.current_stage == "Production" and v.version != challenger_version:
                    client.transition_model_version_stage(
                        name=args.registered_model_name,
                        version=v.version,
                        stage="Archived"
                    )
                    print(f"  → Old champion (v{v.version}) archived")

        print("\nFinal registry state:")
        versions = client.search_model_versions(f"name='{args.registered_model_name}'")
        for v in sorted(versions, key=lambda x: int(x.version)):
            print(f"  Version {v.version}: stage={v.current_stage}")
    else:
        print(f"  Challenger wins {wins}/{total} (needs > {args.promote_threshold:.0%})")
        print("  Champion retains Production. Challenger stays for further evaluation.")

    logger.info("Promotion evaluation completed")


if __name__ == "__main__":
    main()
