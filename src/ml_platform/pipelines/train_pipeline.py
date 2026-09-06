"""End-to-end training pipeline for M1.

Sequence: acquire, validate raw, prepare, validate prepared, split, train,
evaluate, record. Validation failures stop the pipeline before a model is fitted,
so a data problem never becomes a silently degraded model.

The pipeline writes one JSON run record per run under ``artifacts/reports``. That
record is the unit of evidence for the whole project: later milestones compare
candidate records against the production record to decide promotion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_platform import determinism, tracking
from ml_platform.config import Config, load_config
from ml_platform.data import ingestion, preprocessing, validation
from ml_platform.data.splitting import Split, make_splits, subsample
from ml_platform.models.evaluate import EvaluationResult, evaluate_model, lift_over_base_rate
from ml_platform.models.train import TrainedModel, features_and_target, train
from ml_platform.paths import ensure_dir, relative_to_root
from ml_platform.reproducibility import RunContext, new_run_context

LOGGER = logging.getLogger(__name__)

#: Splits the model may be measured on during development. The production stream
#: is deliberately absent: it is reserved for the monitoring and drift work.
DEVELOPMENT_SPLITS = ("train", "validation", "test")


@dataclass
class RunRecord:
    """The full, self-describing result of one training run."""

    context: RunContext
    model_name: str
    feature_set: str
    n_features: int
    train_seconds: float
    splits: list[dict[str, Any]]
    metrics: dict[str, dict[str, Any]]
    validation_reports: list[str]
    dataset: dict[str, Any]
    model_path: str | None = None
    mlflow_run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "model_name": self.model_name,
            "feature_set": self.feature_set,
            "n_features": self.n_features,
            "train_seconds": self.train_seconds,
            "dataset": self.dataset,
            "splits": self.splits,
            "metrics": self.metrics,
            "validation_reports": self.validation_reports,
            "model_path": self.model_path,
            "mlflow_run_id": self.mlflow_run_id,
            "notes": self.notes,
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.context.run_id}-{self.model_name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_prepared_dataset(config: Config, *, nrows: int | None = None) -> tuple[pd.DataFrame, str]:
    """Acquire, validate, clean and label the register.

    Returns the prepared frame and the checksum of the raw file it came from.
    """
    source = config.source
    raw_path = ingestion.acquire(source, config.raw_path)
    checksum = str(source["sha256"])

    LOGGER.info("loading raw register")
    raw = ingestion.load_raw(raw_path, nrows=nrows)
    validation.check_expected_columns(raw)

    expected_rows = int(source.get("expected_rows", 0))
    if nrows is None and expected_rows and len(raw) != expected_rows:
        raise validation.DataValidationError(f"expected {expected_rows} raw rows, found {len(raw)}")

    # The raw schema tolerates the register's known-messy columns; it is checking
    # shape and types, not cleanliness.
    raw_report = validation.validate(raw, validation.RAW_SCHEMA, raise_on_error=False)
    LOGGER.info(raw_report.summary())

    observation_end = pd.Timestamp(config.observation_end)
    prepared = preprocessing.prepare(
        raw,
        config.horizon_months,
        observation_end,
        config.target_column,
        min_term_months=config.min_term_months,
    )
    validation.validate(prepared, validation.PREPARED_SCHEMA)
    validation.assert_label_horizon_elapsed(prepared, config.observation_end)

    LOGGER.info(
        "prepared %s observable rows of %s raw (%.1f%%), positive rate %.4f",
        len(prepared),
        len(raw),
        100 * len(prepared) / max(len(raw), 1),
        prepared[config.target_column].mean(),
    )
    LOGGER.info("row-order fingerprint %s", determinism.row_order_fingerprint(prepared))
    return prepared, checksum


def run_training(
    environment: str = "production",
    model_key: str = "baseline",
    *,
    config: Config | None = None,
    save_model: bool = True,
    nrows: int | None = None,
) -> RunRecord:
    """Train one model end to end and return its run record."""
    cfg = config or load_config(environment)

    # Seeds and thread pinning are applied here so that calling this function
    # directly, not only through the CLI, still produces a reproducible run.
    settings = determinism.configure(cfg.seed, cfg.n_threads)
    unpinned = determinism.verify_thread_pinning(cfg.n_threads)
    if unpinned:
        LOGGER.warning(
            "thread pinning did not take effect for %s; native libraries were "
            "imported before pinning, so bit-exact reproduction is not guaranteed",
            ", ".join(unpinned),
        )

    prepared, checksum = load_prepared_dataset(cfg, nrows=nrows)
    splits = make_splits(prepared, cfg)

    if cfg.sample_fraction < 1.0:
        LOGGER.warning(
            "sampling %.0f%% of each split (environment=%s); results are indicative only",
            100 * cfg.sample_fraction,
            cfg.environment,
        )
        splits = {
            name: subsample(split, cfg.sample_fraction, cfg.seed) for name, split in splits.items()
        }

    spec = dict(cfg.raw[model_key])
    trained = train(spec, splits["train"], cfg.target_column)

    metrics: dict[str, dict[str, Any]] = {}
    for name in DEVELOPMENT_SPLITS:
        result = _evaluate_split(trained, splits[name], cfg, name)
        metrics[name] = {**result.to_dict(), "lift_over_base_rate": lift_over_base_rate(result)}
        LOGGER.info(
            "%-11s AP=%.4f ROC=%.4f Brier=%.4f (skill %+.3f) "
            "top%.0f%%: prec=%.4f rec=%.4f lift=%.2fx | calib=%.2f",
            name,
            result.average_precision,
            result.roc_auc,
            result.brier_score,
            result.brier_skill_score,
            100 * result.review_capacity,
            result.precision_at_capacity,
            result.recall_at_capacity,
            result.lift_at_capacity,
            result.calibration_ratio,
        )

    context = new_run_context(
        config_fingerprint=cfg.fingerprint(),
        environment=cfg.environment,
        seed=cfg.seed,
        data_sha256=checksum,
        determinism=settings,
        prefix=model_key,
    )

    model_path: str | None = None
    if save_model:
        destination = ensure_dir(cfg.model_dir) / f"{context.run_id}-{trained.name}.joblib"
        trained.save(destination)
        # Recorded relative to the project root so records compare across machines.
        model_path = relative_to_root(destination)
        LOGGER.info("saved model to %s", model_path)

    n_features = len(trained.pipeline.named_steps["preprocess"].get_feature_names_out())

    record = RunRecord(
        context=context,
        model_name=trained.name,
        feature_set=trained.feature_set,
        n_features=n_features,
        train_seconds=trained.train_seconds,
        splits=[splits[name].describe(cfg.target_column) for name in cfg.split_names],
        metrics=metrics,
        validation_reports=[
            "raw schema checked",
            "prepared schema enforced",
            "label horizon elapsed for every retained row",
        ],
        dataset={
            "name": cfg.source["name"],
            "sha256": checksum,
            "observation_end": str(cfg.observation_end),
            "label": cfg.raw["label"]["name"],
            "horizon_months": cfg.horizon_months,
            "min_term_months": cfg.min_term_months,
            "observable_rows": len(prepared),
            # Content and row order together. Two runs agreeing here consumed
            # identical data in identical order.
            "row_order_sha256": determinism.row_order_fingerprint(prepared),
        },
        model_path=model_path,
    )

    # Recorded in MLflow before the JSON is written, so the archived record
    # carries the MLflow run id. Tracking cannot fail the run: on any error the
    # id stays None and the JSON record below is written regardless.
    tracking.log_run_record(
        record,
        trained.pipeline,
        cfg,
        model_key=model_key,
        hyperparameters=dict(spec.get("params") or {}),
    )

    saved = record.save(cfg.report_dir)
    LOGGER.info("wrote run record to %s", saved)
    return record


def _evaluate_split(
    trained: TrainedModel, split: Split, config: Config, name: str
) -> EvaluationResult:
    features, target = features_and_target(split, trained.feature_set, config.target_column)
    return evaluate_model(
        trained.pipeline,
        features,
        target,
        split=name,
        review_capacity=config.review_capacity,
    )
