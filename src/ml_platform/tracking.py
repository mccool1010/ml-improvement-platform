"""MLflow experiment tracking.

MLflow **records** the training process; it does not participate in it. The JSON
run record under ``artifacts/reports`` and the locked reference in
``configs/reference.yaml`` remain authoritative, and ``ml_platform reproduce``
never reads MLflow.

The rule that keeps that true: this module accepts a finished
:class:`~ml_platform.pipelines.train_pipeline.RunRecord` and copies values out of
it. **It never computes a metric.** If MLflow and the JSON record ever disagree,
that is a bug in the copying, not an open question about which one is right.

Two consequences follow from the same principle:

* Autologging stays off. It patches scikit-learn's ``fit``, which would put a
  third party inside the determinism contract established at M2.
* A tracking failure is logged and swallowed. Training is the deliverable, and a
  recording tool must not be able to fail it.

A run only reaches this module after training and evaluation have succeeded, so
no MLflow run can represent a failed training run. Should logging itself fail
part way through, the context manager marks the run ``FAILED`` rather than
leaving a half-populated run looking complete.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sklearn.pipeline import Pipeline

    from ml_platform.config import Config
    from ml_platform.pipelines.train_pipeline import RunRecord

LOGGER = logging.getLogger(__name__)

#: MLflow prints an advisory banner about its bundled agent skills on import.
#: It is unrelated to this project and would clutter every training log.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

#: Config values worth recording as parameters, as (parameter name, accessor).
#: Parameters describe the *inputs* to a run and never change once it starts.
_CONFIG_PARAMS: tuple[tuple[str, str], ...] = (
    ("seed", "seed"),
    ("n_threads", "n_threads"),
    ("horizon_months", "horizon_months"),
    ("min_term_months", "min_term_months"),
    ("review_capacity", "review_capacity"),
    ("sample_fraction", "sample_fraction"),
    ("environment", "environment"),
    ("target_column", "target_column"),
)

#: Provenance recorded as tags. Tags are searchable in the MLflow UI, which is
#: how you find "every run on this commit" or "every run on this dataset".
_CONTEXT_TAGS: tuple[str, ...] = (
    "git_revision",
    "git_dirty",
    "config_fingerprint",
    "lockfile_sha256",
    "interpreter",
    "platform",
    "environment",
    "started_at",
)


def build_params(record: RunRecord, config: Config) -> dict[str, Any]:
    """Flatten the run's inputs into MLflow parameters.

    Model hyperparameters are prefixed ``hp_`` so they stay visually distinct
    from platform settings when baseline and candidate sit side by side.
    """
    params: dict[str, Any] = {
        "model_name": record.model_name,
        "feature_set": record.feature_set,
        "n_features": record.n_features,
    }

    for name, attribute in _CONFIG_PARAMS:
        params[name] = getattr(config, attribute)

    for name in config.split_names:
        window = config.window(name)
        params[f"split_{name}"] = f"{window.start}..{window.end}"

    return params


def build_metrics(record: RunRecord) -> dict[str, float]:
    """Flatten every numeric metric the record already holds.

    Names are ``{split}_{metric}``, so ``test_average_precision`` sorts next to
    ``validation_average_precision`` and a chart can plot them together.
    """
    metrics: dict[str, float] = {
        "train_seconds": float(record.train_seconds),
        "n_features": float(record.n_features),
    }

    for split, values in record.metrics.items():
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[f"{split}_{key}"] = float(value)

    for summary in record.splits:
        name = summary.get("name")
        if not name:
            continue
        for key in ("n_rows", "n_positive"):
            if key in summary:
                metrics[f"split_{name}_{key}"] = float(summary[key])

    return metrics


def build_tags(record: RunRecord, model_key: str) -> dict[str, str]:
    """Provenance and identity, as searchable strings."""
    context = record.context.to_dict()

    tags: dict[str, str] = {
        "model_key": model_key,
        "model_name": record.model_name,
        "feature_set": record.feature_set,
        "platform_run_id": record.context.run_id,
        "reproducible": str(record.context.is_reproducible()),
        # Taken from the context, which always carries it. The dataset block is
        # assembled per-pipeline and its shape is not guaranteed.
        "dataset_sha256": record.context.data_sha256,
    }

    for key in _CONTEXT_TAGS:
        if key in context:
            tags[key] = str(context[key])

    dataset = record.dataset
    for key in ("name", "label", "row_order_sha256", "observable_rows", "observation_end"):
        if key in dataset:
            tags[f"dataset_{key}"] = str(dataset[key])

    determinism = record.context.determinism
    tags["sort_kind"] = str(determinism.get("sort_kind", "unknown"))

    for library, version in record.context.libraries.items():
        tags[f"lib_{library}"] = str(version)

    return tags


def _ensure_experiment(mlflow: Any, config: Config) -> None:
    """Select the experiment, creating it with a project-anchored artifact store.

    ``set_experiment`` alone would let MLflow default the artifact location to a
    directory relative to the working directory, which would break the M2 rule
    that nothing depends on where a command was launched from.
    """
    existing = mlflow.get_experiment_by_name(config.experiment_name)
    if existing is None:
        mlflow.create_experiment(config.experiment_name, artifact_location=config.artifact_uri)
    mlflow.set_experiment(config.experiment_name)


def log_run_record(
    record: RunRecord,
    model: Pipeline | None,
    config: Config,
    *,
    model_key: str = "baseline",
    hyperparameters: dict[str, Any] | None = None,
) -> str | None:
    """Record a finished run in MLflow. Returns the MLflow run id, or None.

    Returns None when tracking is disabled or when logging fails. Neither case
    is an error for the caller: the JSON run record is the deliverable.
    """
    if not config.tracking_enabled:
        LOGGER.debug("tracking disabled; skipping MLflow")
        return None

    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        _ensure_experiment(mlflow, config)

        params = build_params(record, config)
        for name, value in (hyperparameters or {}).items():
            params[f"hp_{name}"] = value

        with mlflow.start_run(run_name=record.context.run_id) as run:
            # Set before serialising, so the archived record carries the link
            # back to this MLflow run.
            record.mlflow_run_id = run.info.run_id

            mlflow.log_params(params)
            mlflow.log_metrics(build_metrics(record))
            mlflow.set_tags(build_tags(record, model_key))
            mlflow.log_dict(record.to_dict(), "run_record.json")

            if model is not None and config.log_model:
                import mlflow.sklearn

                # MLflow 3 defaults to the skops serialiser, which refuses to
                # write a pipeline referencing numpy.dtype. Cloudpickle matches
                # what the project already uses for its own joblib artifacts.
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    serialization_format="cloudpickle",
                )

            LOGGER.info(
                "logged MLflow run %s to experiment %r",
                run.info.run_id,
                config.experiment_name,
            )
            return str(run.info.run_id)

    except Exception:
        LOGGER.warning("MLflow tracking failed; the run record is unaffected", exc_info=True)
        record.mlflow_run_id = None
        return None
