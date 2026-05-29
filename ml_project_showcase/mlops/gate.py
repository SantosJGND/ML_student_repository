import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_logger, MLFLOW_TRACKING_URI, registry_name
from evaluation.gate import check_gates
from mlops.promotion import transition_to_staging

logger = get_logger(__name__)

DEFAULT_GATES = {"accuracy": 0.95, "f1": 0.90}


def evaluate_and_promote(model_type, metrics, run_id):
    passed, details = check_gates(metrics, DEFAULT_GATES)
    logger.info(f"Gate check for {model_type}: {'PASSED' if passed else 'FAILED'}")
    for key, d in details.items():
        logger.info(f"  {key}: {d['value']:.4f} >= {d['threshold']:.4f} = {d['passed']}")
    if passed:
        name = registry_name(model_type)
        transition_to_staging(name, run_id)
        logger.info(f"{model_type} promoted to Staging")
    else:
        logger.warning(f"{model_type} did NOT pass gates — not promoted")
    return passed, details


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model gates and promote")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--accuracy", type=float)
    parser.add_argument("--f1", type=float)
    args = parser.parse_args()

    metrics = {}
    if args.accuracy is not None:
        metrics["accuracy"] = args.accuracy
    if args.f1 is not None:
        metrics["f1"] = args.f1

    if not metrics:
        print("No metrics provided. Use --accuracy and/or --f1.")
        sys.exit(1)

    passed, details = evaluate_and_promote(args.model_type, metrics, args.run_id)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
