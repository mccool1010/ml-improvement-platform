"""MLflow Model Registry.

Registration is the *consequence* of a promotion decision, never the decision
itself. :mod:`ml_platform.promotion.gates` decides; this module writes down what
was decided. A rejected candidate is left as an ordinary MLflow run, with its
model still logged as an artifact, and never becomes a registered version.

Two identities are kept joined so a registered version can always be traced
back: the MLflow run that produced the model, and the platform's own run id from
the JSON record. Both are stored as version tags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.config import Config
    from ml_platform.pipelines.train_pipeline import RunRecord
    from ml_platform.promotion.gates import GateReport

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionModel:
    """Whatever is currently acting as production, and how it was identified."""

    source: str
    name: str
    version: str | None = None
    run_id: str | None = None
    platform_run_id: str | None = None

    def describe(self) -> str:
        if self.source == "registry":
            return f"{self.name} v{self.version} (registry alias)"
        return f"{self.name} (bootstrap: nothing registered yet)"


def _client(config: Config) -> Any:
    """An MLflow client pointed at the project's store."""
    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)
    return mlflow.MlflowClient()


def resolve_production(config: Config) -> ProductionModel | None:
    """Find the registered production model, or None if there is not one yet.

    Identification is by the configured registry alias, never by recency. The
    newest run is not production, and treating it as such would let any accident
    become the incumbent.
    """
    try:
        client = _client(config)
        version = client.get_model_version_by_alias(
            config.registered_model_name, config.production_alias
        )
    except Exception:
        LOGGER.info(
            "no %r alias on registered model %r; production will be bootstrapped",
            config.production_alias,
            config.registered_model_name,
        )
        return None

    return ProductionModel(
        source="registry",
        name=config.registered_model_name,
        version=str(version.version),
        run_id=str(version.run_id),
        platform_run_id=dict(version.tags or {}).get("platform_run_id"),
    )


def register_candidate(
    config: Config,
    record: RunRecord,
    report: GateReport,
) -> tuple[str | None, str | None]:
    """Register the candidate and move the production alias to it.

    Returns ``(model_name, version)``, or ``(None, None)`` if registration could
    not complete. Refuses outright when the gates did not pass, so this function
    cannot be the way a rejected model reaches the registry.
    """
    if not report.promote:
        raise ValueError(
            "refusing to register a candidate that failed "
            f"{len(report.failures)} mandatory gate(s): "
            f"{', '.join(g.name for g in report.failures)}"
        )
    if not record.mlflow_run_id:
        LOGGER.warning("candidate has no MLflow run; nothing to register")
        return None, None

    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        client = mlflow.MlflowClient()

        # The model artifact was logged with cloudpickle by the tracking layer;
        # registering the URI reuses that artifact rather than reserialising.
        source_uri = f"runs:/{record.mlflow_run_id}/model"
        version = mlflow.register_model(
            model_uri=source_uri,
            name=config.registered_model_name,
            tags=_version_tags(record, report),
        )

        client.set_registered_model_alias(
            config.registered_model_name, config.production_alias, version.version
        )
        LOGGER.info(
            "registered %s v%s and moved alias %r",
            config.registered_model_name,
            version.version,
            config.production_alias,
        )
        return config.registered_model_name, str(version.version)

    except Exception:
        LOGGER.warning("model registration failed; the gate report stands", exc_info=True)
        return None, None


def _version_tags(record: RunRecord, report: GateReport) -> dict[str, str]:
    """Traceability and provenance, carried on the registered version."""
    context = record.context
    metrics = record.metrics.get(report.decision_split, {})

    tags: dict[str, str] = {
        "platform_run_id": context.run_id,
        "mlflow_run_id": str(record.mlflow_run_id),
        "model_name": record.model_name,
        "feature_set": record.feature_set,
        "git_revision": context.git_revision,
        "config_fingerprint": context.config_fingerprint,
        "dataset_sha256": context.data_sha256,
        "lockfile_sha256": context.lockfile_sha256,
        "decision_split": report.decision_split,
        "gates_passed": str(sum(1 for g in report.gates if g.passed)),
        "gates_total": str(len(report.gates)),
        "promoted_over": report.production_name,
    }
    for key in ("row_order_sha256", "label", "horizon_months"):
        if key in record.dataset:
            tags[f"dataset_{key}"] = str(record.dataset[key])
    for metric in ("average_precision", "roc_auc", "brier_score", "recall_at_capacity"):
        if metric in metrics:
            tags[f"{report.decision_split}_{metric}"] = str(metrics[metric])
    return tags
