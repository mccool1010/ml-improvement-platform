"""Validate the dataset without training anything.

    python scripts/validate_data.py

Exits non-zero on a schema failure, so CI can use it as a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_platform.config import load_config
from ml_platform.data.splitting import make_splits
from ml_platform.data.validation import DataValidationError
from ml_platform.pipelines.train_pipeline import load_prepared_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the SBA dataset and report splits.")
    parser.add_argument("--environment", default="production")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    config = load_config(args.environment)

    try:
        prepared, checksum = load_prepared_dataset(config)
        splits = make_splits(prepared, config)
    except DataValidationError as exc:
        print(f"VALIDATION FAILED\n{exc}", file=sys.stderr)
        return 1

    summary = {
        "dataset_sha256": checksum,
        "observable_rows": len(prepared),
        "overall_positive_rate": round(float(prepared[config.target_column].mean()), 6),
        "splits": [splits[name].describe(config.target_column) for name in config.split_names],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
