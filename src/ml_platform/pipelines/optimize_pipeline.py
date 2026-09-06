"""Optimisation pipeline for M5.

Runs a bounded Optuna search over the candidate model, then answers the only
question that matters: **did searching produce a measurably better model than the
baseline already in place?**

The comparison deliberately goes through :func:`run_training`, so the tuned
candidate and the baseline are measured by the same code, on the same splits,
and both leave an ordinary ``RunRecord`` behind. No promotion decision is made
here. Deciding whether an improvement is large enough to deploy is M6's job.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ml_platform import determinism, tracking
from ml_platform.config import Config, load_config
from ml_platform.data.splitting import make_splits, subsample
from ml_platform.models.optimize import StudyResult, run_study
from ml_platform.paths import ensure_dir
from ml_platform.pipelines.train_pipeline import RunRecord, load_prepared_dataset, run_training

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """The search, plus the head-to-head it produced."""

    study: StudyResult
    baseline: RunRecord
    tuned: RunRecord
    parent_run_id: str | None

    def comparison(self, metric: str = "average_precision") -> dict[str, Any]:
        """Baseline against tuned candidate on the held-out test split."""
        baseline_value = float(self.baseline.metrics["test"][metric])
        tuned_value = float(self.tuned.metrics["test"][metric])
        return {
            "metric": metric,
            "split": "test",
            "baseline": baseline_value,
            "tuned_candidate": tuned_value,
            "absolute_improvement": round(tuned_value - baseline_value, 6),
            "improved": tuned_value > baseline_value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "study": self.study.to_dict(),
            "mlflow_parent_run_id": self.parent_run_id,
            "baseline_run_id": self.baseline.context.run_id,
            "tuned_run_id": self.tuned.context.run_id,
            "tuned_mlflow_run_id": self.tuned.mlflow_run_id,
            "comparison": self.comparison(),
            "test_metrics": {
                "baseline": self.baseline.metrics["test"],
                "tuned_candidate": self.tuned.metrics["test"],
            },
        }


def run_optimization(
    environment: str = "production",
    *,
    config: Config | None = None,
    n_trials: int | None = None,
    save_model: bool = True,
    nrows: int | None = None,
) -> OptimizationResult:
    """Search, then measure the winner against the baseline."""
    cfg = config or load_config(environment)
    determinism.configure(cfg.seed, cfg.n_threads)

    prepared, _ = load_prepared_dataset(cfg, nrows=nrows)
    splits = make_splits(prepared, cfg)
    if cfg.sample_fraction < 1.0:
        splits = {
            name: subsample(split, cfg.sample_fraction, cfg.seed) for name, split in splits.items()
        }

    study_name = f"study-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    base_spec = dict(cfg.raw["candidate"])

    with tracking.StudyTracker(cfg, study_name) as tracker:
        tracker.log_study_setup(cfg)
        study = run_study(
            cfg,
            splits,
            base_spec,
            study_name=study_name,
            n_trials=n_trials,
            sink=tracker if tracker.active else None,
        )
        tracker.log_best(study)
        parent_run_id = tracker.parent_run_id

    LOGGER.info(
        "study complete: %d/%d trials succeeded, best %s=%s on %s",
        study.n_completed,
        study.n_trials,
        study.objective_metric,
        "n/a" if study.best_value is None else f"{study.best_value:.6f}",
        study.objective_split,
    )

    if not study.best_params:
        raise RuntimeError("no trial completed successfully; there is no candidate to evaluate")

    # Both through the same machinery, so the comparison is like for like.
    LOGGER.info("training the baseline for comparison")
    baseline = run_training(config=cfg, model_key="baseline", save_model=False, nrows=nrows)

    LOGGER.info("training the tuned candidate")
    tuned = run_training(
        config=cfg,
        model_key="candidate",
        save_model=save_model,
        nrows=nrows,
        param_overrides=study.best_params,
    )

    result = OptimizationResult(
        study=study, baseline=baseline, tuned=tuned, parent_run_id=parent_run_id
    )

    destination = ensure_dir(cfg.benchmark_dir) / f"optimization-{study_name}.json"
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("wrote optimisation report to %s", destination)
    return result
