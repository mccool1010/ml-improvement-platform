"""Train a model and write a run record.

python scripts/train.py --environment production --model baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_platform.pipelines.train_pipeline import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a model on the SBA register.")
    parser.add_argument(
        "--environment", default="production", choices=["development", "production"]
    )
    parser.add_argument("--model", default="baseline", choices=["baseline", "candidate"])
    parser.add_argument("--no-save", action="store_true", help="do not write the model artifact")
    parser.add_argument("--nrows", type=int, default=None, help="read only N raw rows (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    record = run_training(
        environment=args.environment,
        model_key=args.model,
        save_model=not args.no_save,
        nrows=args.nrows,
    )

    test = record.metrics["test"]
    print(
        f"\n{record.model_name}: test AP={test['average_precision']:.4f} "
        f"ROC={test['roc_auc']:.4f} lift={test['lift_over_base_rate']:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
