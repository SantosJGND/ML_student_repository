#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Register model in MLflow')
    parser.add_argument('--model-uri', type=str, required=True,
                        help='MLflow model URI to register')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name for registered model')
    args = parser.parse_args()
    logger.info("Registration script completed (TODO: implement logic)")


if __name__ == "__main__":
    main()
