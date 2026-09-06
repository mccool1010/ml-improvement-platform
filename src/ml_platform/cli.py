"""Command line entry point.

    python -m ml_platform <command>

One surface for every operation, so there is a single documented path to
reproduce a result rather than a collection of ad-hoc scripts.

The import order in this module is deliberate. Native thread pools read their
environment when the library first loads, so :func:`ml_platform.determinism.pin_threads`
must run before numpy, scipy or scikit-learn are imported. Only the configuration
and determinism modules are imported at module scope, and both are pure Python.
Everything heavy is imported inside the command functions, after pinning.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ml_platform.config import Config, load_config
from ml_platform.determinism import pin_threads

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")


def _bootstrap(environment: str) -> Config:
    """Load configuration and pin native threads before anything heavy imports."""
    config = load_config(environment)
    pin_threads(config.n_threads)
    return config


def command_download(args: argparse.Namespace) -> int:
    """Download the register and verify its checksum."""
    config = _bootstrap(args.environment)
    from ml_platform.data import ingestion
    from ml_platform.paths import relative_to_root

    path = ingestion.acquire(config.source, config.raw_path, force=args.force)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"verified {relative_to_root(path)} ({size_mb:.1f} MB)")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    """Validate the dataset and report split composition, without training."""
    import json

    config = _bootstrap(args.environment)
    from ml_platform.data.splitting import make_splits
    from ml_platform.data.validation import DataValidationError
    from ml_platform.pipelines.train_pipeline import load_prepared_dataset

    try:
        prepared, checksum = load_prepared_dataset(config)
        splits = make_splits(prepared, config)
    except DataValidationError as exc:
        print(f"VALIDATION FAILED\n{exc}", file=sys.stderr)
        return 1

    target = config.target_column
    print(
        json.dumps(
            {
                "dataset_sha256": checksum,
                "observable_rows": len(prepared),
                "overall_positive_rate": round(float(prepared[target].mean()), 6),
                "splits": [splits[name].describe(target) for name in config.split_names],
            },
            indent=2,
        )
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    """Train one model and write its run record."""
    _bootstrap(args.environment)
    from ml_platform.pipelines.train_pipeline import run_training

    record = run_training(
        environment=args.environment,
        model_key=args.model,
        save_model=not args.no_save,
        nrows=args.nrows,
    )
    test = record.metrics["test"]
    print(
        f"\n{record.model_name}: test AP={test['average_precision']:.4f} "
        f"ROC={test['roc_auc']:.4f} recall@10%={test['recall_at_capacity']:.4f} "
        f"lift={test['lift_at_capacity']:.2f}x"
    )
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    """Retrain both models and compare against the locked reference metrics."""
    _bootstrap(args.environment)
    from ml_platform.pipelines.reproduce import reproduce

    result = reproduce(
        environment=args.environment,
        profile=args.profile,
        save_models=args.save_models,
    )

    print()
    print(result.summary())
    provenance = result.provenance
    print(f"  commit          {provenance['git_revision'][:12]} (dirty={provenance['git_dirty']})")
    print(f"  config          {provenance['config_fingerprint']}")
    print(f"  dataset         {provenance['dataset_sha256'][:12]}")
    print(f"  lockfile        {provenance['lockfile_sha256']}")
    print(f"  interpreter     {provenance['interpreter']} on {provenance['platform']}")
    print(f"  threads         {provenance['determinism']['n_threads']}")

    for mismatch in result.split_mismatches:
        print(f"  SPLIT MISMATCH  {mismatch}", file=sys.stderr)
    for deviation in result.deviations:
        print(f"  OUT OF BOUNDS   {deviation.describe()}", file=sys.stderr)

    if not result.passed:
        print("\nreproduction FAILED", file=sys.stderr)
        return 1
    print("\nreproduction PASSED: every metric matches the locked M1 reference")
    return 0


def command_optimize(args: argparse.Namespace) -> int:
    """Search the candidate's hyperparameters and compare the winner to baseline."""
    _bootstrap(args.environment)
    from ml_platform.pipelines.optimize_pipeline import run_optimization

    result = run_optimization(
        environment=args.environment,
        n_trials=args.trials,
        save_model=not args.no_save,
        nrows=args.nrows,
    )

    study = result.study
    comparison = result.comparison()
    print()
    print(f"study {study.study_name}: {study.n_completed} completed, {study.n_failed} failed")
    print(f"  objective        {study.objective_metric} on {study.objective_split}")
    print(f"  best trial       #{study.best_trial_number}  value={study.best_value:.6f}")
    print(f"  untuned value    {study.baseline_value:.6f}")
    print(f"  search gain      {study.improvement_over_untuned:+.6f}")
    print(f"  best params      {study.best_params}")
    print(f"  mlflow study run {result.parent_run_id}")
    print()
    print(f"held-out test {comparison['metric']}:")
    print(f"  baseline         {comparison['baseline']:.6f}")
    print(f"  tuned candidate  {comparison['tuned_candidate']:.6f}")
    print(f"  improvement      {comparison['absolute_improvement']:+.6f}")
    print(f"  improved         {comparison['improved']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml_platform",
        description="ML improvement platform: data, training and reproducibility.",
    )
    parser.add_argument(
        "--environment",
        default="production",
        choices=["development", "production"],
        help="configuration overlay to apply (default: production)",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and verify the raw dataset")
    download.add_argument("--force", action="store_true", help="re-download even if present")
    download.set_defaults(handler=command_download)

    validate = subparsers.add_parser("validate", help="validate data and report splits")
    validate.set_defaults(handler=command_validate)

    train = subparsers.add_parser("train", help="train a model and write a run record")
    train.add_argument("--model", default="baseline", choices=["baseline", "candidate"])
    train.add_argument("--no-save", action="store_true", help="skip writing the model artifact")
    train.add_argument("--nrows", type=int, default=None, help="read only N raw rows (smoke test)")
    train.set_defaults(handler=command_train)

    optimize = subparsers.add_parser(
        "optimize", help="run a hyperparameter search and compare it to the baseline"
    )
    optimize.add_argument("--trials", type=int, default=None, help="override the configured budget")
    optimize.add_argument("--no-save", action="store_true", help="skip writing the model artifact")
    optimize.add_argument(
        "--nrows", type=int, default=None, help="read only N raw rows (smoke test)"
    )
    optimize.set_defaults(handler=command_optimize)

    repro = subparsers.add_parser(
        "reproduce", help="verify the run against locked reference metrics"
    )
    repro.add_argument(
        "--profile",
        default="strict",
        choices=["strict", "portable"],
        help="tolerance profile: strict for the reference platform, portable elsewhere",
    )
    repro.add_argument("--save-models", action="store_true", help="also write model artifacts")
    repro.set_defaults(handler=command_reproduce)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
