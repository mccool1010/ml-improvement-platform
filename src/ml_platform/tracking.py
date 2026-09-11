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
from typing import TYPE_CHECKING, Any, Literal

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


def _ensure_experiment(mlflow: Any, config: Config, name: str | None = None) -> None:
    """Select the experiment, creating it with an appropriate artifact store.

    Against a local store, the artifact location is set explicitly. ``set_experiment``
    alone would let MLflow default it to a directory relative to the working
    directory, which would break the M2 rule that nothing depends on where a
    command was launched from.

    Against a tracking *server*, no location is passed. The location is written
    into the experiment record, so a path chosen by whichever client happened to
    create the experiment would be recorded for every other client -- which is
    precisely how an absolute Windows path came to be baked into the local store
    and left the artifacts unreachable from a container. The server assigns its
    own root and serves artifacts over HTTP instead.
    """
    experiment = name or config.experiment_name
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=config.artifact_uri)
    mlflow.set_experiment(experiment)


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


class StudyTracker:
    """Parent and nested MLflow runs for one Optuna study.

    Every method swallows its own failures. A search must not fail because a
    recorder did, and Optuna's decision about the best trial never consults this
    object: it only receives what already happened.

    The parent run represents the study; one nested run is created per trial.
    A trial that failed is ended with status ``FAILED``, so a study's run list
    cannot present a failed trial as a completed one.
    """

    def __init__(self, config: Config, study_name: str) -> None:
        self._config = config
        self._study_name = study_name
        self._mlflow: Any | None = None
        self.parent_run_id: str | None = None

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def __enter__(self) -> StudyTracker:
        if not self._config.tracking_enabled:
            LOGGER.debug("tracking disabled; the study will not be recorded")
            return self
        try:
            import mlflow

            mlflow.set_tracking_uri(self._config.tracking_uri)
            _ensure_experiment(mlflow, self._config, self._config.optimization_experiment_name)
            run = mlflow.start_run(run_name=self._study_name)
            self._mlflow = mlflow
            self.parent_run_id = str(run.info.run_id)
            mlflow.set_tags({"study": "true", "study_name": self._study_name})
            LOGGER.info("MLflow study run %s", self.parent_run_id)
        except Exception:
            LOGGER.warning("could not start the MLflow study run", exc_info=True)
            self._mlflow = None
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        if self._mlflow is not None:
            try:
                self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            except Exception:
                LOGGER.warning("could not close the MLflow study run", exc_info=True)
        return False

    def log_study_setup(self, config: Config) -> None:
        """Record what the search was asked to do, before any trial runs."""
        if self._mlflow is None:
            return
        try:
            params: dict[str, Any] = {
                "n_trials": config.n_trials,
                "sampler": "TPESampler",
                "sampler_seed": config.sampler_seed,
                "objective_metric": config.objective_metric,
                "objective_split": config.objective_split,
                "direction": config.objective_direction,
                "seed": config.seed,
            }
            for name, spec in config.search_space.items():
                params[f"space_{name}"] = str(spec)
            self._mlflow.log_params(params)
        except Exception:
            LOGGER.warning("could not log study setup", exc_info=True)

    def log_trial(
        self,
        number: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        *,
        failed: bool = False,
    ) -> None:
        """Record one trial as a nested run under the study."""
        if self._mlflow is None:
            return
        try:
            self._mlflow.start_run(run_name=f"trial-{number:03d}", nested=True)
            try:
                self._mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
                if metrics:
                    self._mlflow.log_metrics(metrics)
                self._mlflow.set_tags(
                    {
                        "trial_number": str(number),
                        "trial_state": "FAIL" if failed else "COMPLETE",
                        "study_name": self._study_name,
                    }
                )
            finally:
                self._mlflow.end_run(status="FAILED" if failed else "FINISHED")
        except Exception:
            LOGGER.warning("could not log trial %s", number, exc_info=True)

    def log_best(self, result: Any) -> None:
        """Record the winning trial on the parent run.

        Optuna selected it; this only writes it down.
        """
        if self._mlflow is None:
            return
        try:
            metrics: dict[str, float] = {
                "n_completed_trials": float(result.n_completed),
                "n_failed_trials": float(result.n_failed),
            }
            if result.best_value is not None:
                metrics["best_objective_value"] = float(result.best_value)
                metrics["best_trial_number"] = float(result.best_trial_number or 0)
            if result.baseline_value is not None:
                metrics["untuned_objective_value"] = float(result.baseline_value)
            if result.improvement_over_untuned is not None:
                metrics["improvement_over_untuned"] = float(result.improvement_over_untuned)

            self._mlflow.log_metrics(metrics)
            if result.best_params:
                self._mlflow.log_params({f"best_{k}": v for k, v in result.best_params.items()})
            self._mlflow.log_dict(result.to_dict(), "study.json")
        except Exception:
            LOGGER.warning("could not log the study result", exc_info=True)
