"""Candidate against production.

Produces the two metric sets a gate evaluation needs, and is explicit about
where each one came from.

Production is identified in one of exactly two ways, in order:

1. the model carrying the configured registry alias;
2. failing that, the explicitly configured ``bootstrap_model``, trained fresh.

There is deliberately no third way. In particular the newest MLflow run is never
treated as production: that would let any accidental run become the incumbent
that everything else is measured against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ml_platform.config import Config
from ml_platform.pipelines.train_pipeline import RunRecord, run_training
from ml_platform.promotion.registry import ProductionModel, resolve_production

LOGGER = logging.getLogger(__name__)


@dataclass
class Comparison:
    """Two models, their decision-split metrics, and their provenance."""

    decision_split: str
    candidate: RunRecord
    production_metrics: dict[str, Any]
    production: ProductionModel
    production_record: RunRecord | None = None

    @property
    def candidate_metrics(self) -> dict[str, Any]:
        return dict(self.candidate.metrics[self.decision_split])

    @property
    def context(self) -> dict[str, Any]:
        """Provenance the reproducibility gate reads."""
        context = self.candidate.context
        return {
            "reproducible": context.is_reproducible(),
            "git_revision": context.git_revision,
            "git_dirty": context.git_dirty,
            "lockfile_sha256": context.lockfile_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_split": self.decision_split,
            "candidate": {
                "run_id": self.candidate.context.run_id,
                "mlflow_run_id": self.candidate.mlflow_run_id,
                "model_name": self.candidate.model_name,
                "metrics": self.candidate_metrics,
            },
            "production": {
                "source": self.production.source,
                "described": self.production.describe(),
                "name": self.production.name,
                "version": self.production.version,
                "mlflow_run_id": self.production.run_id,
                "platform_run_id": self.production.platform_run_id,
                "metrics": self.production_metrics,
            },
        }


class ProductionResolutionError(RuntimeError):
    """Raised when no production model can be established at all."""


def _metrics_from_registry(
    config: Config, production: ProductionModel, split: str
) -> dict[str, Any] | None:
    """Read the incumbent's decision-split metrics from its own MLflow run.

    Metrics logged by the tracking layer are namespaced ``{split}_{metric}``,
    which is undone here. These are the numbers the incumbent actually recorded
    when it was promoted, not a re-derivation.
    """
    if not production.run_id:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        run = mlflow.MlflowClient().get_run(production.run_id)
    except Exception:
        LOGGER.warning("could not read the production run %s", production.run_id, exc_info=True)
        return None

    prefix = f"{split}_"
    metrics = {
        name[len(prefix) :]: value
        for name, value in run.data.metrics.items()
        if name.startswith(prefix)
    }
    return metrics or None


def build_comparison(
    config: Config,
    candidate: RunRecord,
    *,
    nrows: int | None = None,
) -> Comparison:
    """Pair the candidate with whatever is currently production."""
    split = config.decision_split
    if split not in candidate.metrics:
        raise ProductionResolutionError(
            f"the candidate has no {split!r} metrics; it cannot be judged"
        )

    production = resolve_production(config)
    if production is not None:
        metrics = _metrics_from_registry(config, production, split)
        if metrics:
            LOGGER.info("production is %s", production.describe())
            return Comparison(
                decision_split=split,
                candidate=candidate,
                production_metrics=metrics,
                production=production,
            )
        LOGGER.warning(
            "the registered production model has no %s metrics recorded; "
            "falling back to the configured bootstrap model",
            split,
        )

    # Nothing registered yet. Train the explicitly configured stand-in rather
    # than adopting whatever ran most recently.
    LOGGER.info("bootstrapping production from the configured %r model", config.bootstrap_model)
    record = run_training(
        config=config, model_key=config.bootstrap_model, save_model=False, nrows=nrows
    )
    return Comparison(
        decision_split=split,
        candidate=candidate,
        production_metrics=dict(record.metrics[split]),
        production=ProductionModel(source="bootstrap", name=record.model_name),
        production_record=record,
    )
