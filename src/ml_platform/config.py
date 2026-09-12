"""Configuration loading.

Configuration is layered: ``base.yaml`` always applies, and an environment file
(``development``/``production``) overrides it. Every run records the resolved
configuration and its hash so a result can be tied back to the exact settings
that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ml_platform.paths import project_root, resolve

CONFIG_DIR = project_root() / "configs"

#: Environment variable that overrides the configured tracking backend. Named
#: for MLflow's own convention so a deployment sets one familiar thing.
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"

#: Address of the KServe model tier. Its presence is what decides where scoring
#: happens: set, the API calls the InferenceService; unset, it loads the model
#: itself. One switch, so the two cannot be configured into contradiction.
ENV_PREDICTOR_URL = "ML_PLATFORM_PREDICTOR_URL"

#: OTLP trace collector. OpenTelemetry's own standard variable name.
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"

#: The canary model tier, and its share of traffic. Both come from the
#: deployment so every API replica agrees on them by construction.
ENV_CANARY_URL = "ML_PLATFORM_CANARY_URL"
ENV_CANARY_TRAFFIC = "ML_PLATFORM_CANARY_TRAFFIC_PERCENT"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class DateWindow:
    """A half-open-in-spirit but inclusive date range used for time-ordered splits."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"window start {self.start} is after end {self.end}")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DateWindow:
        return cls(start=_as_date(raw["start"]), end=_as_date(raw["end"]))


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass(frozen=True)
class Config:
    """Resolved project configuration."""

    raw: dict[str, Any]
    environment: str

    # --- convenience accessors -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def sample_fraction(self) -> float:
        return float(self.raw.get("sample_fraction", 1.0))

    @property
    def source(self) -> dict[str, Any]:
        return dict(self.raw["data"]["source"])

    @property
    def raw_path(self) -> Path:
        return resolve(str(self.raw["data"]["raw_dir"]), str(self.source["filename"]))

    @property
    def processed_dir(self) -> Path:
        return resolve(str(self.raw["data"]["processed_dir"]))

    @property
    def interim_dir(self) -> Path:
        return resolve(str(self.raw["data"]["interim_dir"]))

    @property
    def report_dir(self) -> Path:
        return resolve(str(self.raw["evaluation"]["report_dir"]))

    @property
    def model_dir(self) -> Path:
        return resolve(str(self.raw["artifacts"]["model_dir"]))

    @property
    def benchmark_dir(self) -> Path:
        return resolve(str(self.raw["artifacts"]["benchmark_dir"]))

    @property
    def n_threads(self) -> int:
        """Native thread-pool size. Pinned so parallel reductions stay ordered."""
        return int(self.raw["determinism"]["n_threads"])

    @property
    def _tracking(self) -> dict[str, Any]:
        """Tracking settings, or an empty mapping when the block is absent."""
        return dict(self.raw.get("tracking") or {})

    @property
    def tracking_enabled(self) -> bool:
        """Whether to record this run in MLflow. Absent configuration means no."""
        return bool(self._tracking.get("enabled", False))

    @property
    def experiment_name(self) -> str:
        return str(self._tracking.get("experiment_name", "default"))

    @property
    def log_model(self) -> bool:
        return bool(self._tracking.get("log_model", False))

    @property
    def tracking_uri(self) -> str:
        """MLflow backend store, with any relative path anchored to the project.

        A ``sqlite:///`` URI carrying a relative path is rewritten to an absolute
        one, so the store does not move with the working directory. Anything
        else, including a remote server, passes through untouched.

        ``MLFLOW_TRACKING_URI`` in the environment takes precedence. That is how
        a deployment supplies the address of its own tracking server without an
        environment-specific value being baked into the image, and it is the
        variable the MLflow ecosystem already uses. Without this the variable
        would be silently ignored, because every call site sets the URI from
        configuration explicitly.
        """
        raw = os.environ.get(ENV_TRACKING_URI) or str(
            self._tracking.get("backend_uri", "sqlite:///mlflow.db")
        )
        prefix = "sqlite:///"
        if raw.startswith(prefix):
            target = raw[len(prefix) :]
            if not Path(target).is_absolute():
                return prefix + resolve(target).as_posix()
        return raw

    @property
    def tracking_is_remote(self) -> bool:
        """Whether tracking goes to a server rather than a local store.

        A remote server owns its own artifact root and serves artifacts over
        HTTP, so the client must not impose a local path on it. That is the
        distinction that makes the store portable: a path chosen here is
        recorded in the database and becomes meaningless on any other machine.
        """
        return self.tracking_uri.startswith(("http://", "https://"))

    @property
    def artifact_uri(self) -> str | None:
        """Where MLflow stores run artifacts, as a file URI under the project.

        ``None`` against a remote tracking server, which assigns its own
        artifact location. See :attr:`tracking_is_remote`.
        """
        if self.tracking_is_remote:
            return None
        return resolve(str(self._tracking.get("artifact_dir", "mlartifacts"))).as_uri()

    # --- serving (M11) ----------------------------------------------------
    @property
    def _serving(self) -> dict[str, Any]:
        return dict(self.raw.get("serving") or {})

    @property
    def predictor_url(self) -> str | None:
        """Base URL of the KServe predictor, or ``None`` to score in process.

        ADR-002 makes KServe the canonical serving path in Kubernetes and leaves
        FastAPI as the application tier in front of it. Locally there is no
        cluster and no InferenceService, so the API loads the model itself. This
        is the single value that distinguishes those two, and it is set by the
        deployment rather than chosen by the code.
        """
        configured = os.environ.get(ENV_PREDICTOR_URL) or self._serving.get("predictor_url")
        return str(configured) if configured else None

    @property
    def predictor_model_name(self) -> str:
        """Model name in the predictor's V1 paths; the InferenceService's name."""
        return str(self._serving.get("predictor_model_name", "sba-loan-default"))

    @property
    def predictor_timeout_seconds(self) -> float:
        return float(self._serving.get("predictor_timeout_seconds", 10.0))

    # --- observability (M12) ----------------------------------------------
    @property
    def otlp_endpoint(self) -> str | None:
        """OTLP/HTTP collector for traces, or ``None`` to run without tracing.

        Named for OpenTelemetry's own convention, so a deployment sets the
        variable the ecosystem already uses. Absent -- locally and under test --
        there is no collector to export to and tracing stays off, which is the
        right default rather than a degraded one.
        """
        configured = os.environ.get(ENV_OTLP_ENDPOINT) or self._observability.get("otlp_endpoint")
        return str(configured) if configured else None

    @property
    def _observability(self) -> dict[str, Any]:
        return dict(self.raw.get("observability") or {})

    # --- canary (M14) ------------------------------------------------------
    @property
    def _canary(self) -> dict[str, Any]:
        return dict(self.raw.get("canary") or {})

    @property
    def canary_alias(self) -> str:
        """Registry alias a candidate holds *while* it is being canaried.

        Separate from the production alias on purpose: a candidate under test
        must be identifiable and loadable without being production. Configuring
        them to the same string is refused here rather than discovered later:
        it would make starting a canary move production immediately, which is
        the one thing the whole arrangement exists to prevent.
        """
        alias = str(self._canary.get("alias", "canary"))
        if alias == self.production_alias:
            from ml_platform.serving.canary import CanaryError

            raise CanaryError(
                f"the canary alias and the production alias are both {alias!r}. "
                "Starting a canary would move production immediately, before any "
                "traffic had reached the candidate."
            )
        return alias

    @property
    def canary_model_name(self) -> str:
        """InferenceService name for the canary, in the predictor's V1 paths."""
        return str(self._canary.get("model_name", "sba-loan-default-canary"))

    @property
    def canary_predictor_url(self) -> str | None:
        configured = os.environ.get(ENV_CANARY_URL) or self._canary.get("predictor_url")
        return str(configured) if configured else None

    @property
    def canary_traffic_percent(self) -> float:
        """Share of traffic the canary receives, set by the deployment.

        Validated here as well as in the router, because a bad value in a
        ConfigMap should fail the pod at startup rather than after it has begun
        sending every request to an unproven model.
        """
        configured = os.environ.get(ENV_CANARY_TRAFFIC)
        percent = (
            float(configured)
            if configured is not None
            else float(self._canary.get("traffic_percent", 0.0))
        )
        from ml_platform.serving.canary import _validate_traffic

        _validate_traffic(percent)
        return percent

    @property
    def canary_enabled(self) -> bool:
        """A canary runs only when it has somewhere to send traffic."""
        return bool(self.canary_predictor_url) and self.canary_traffic_percent > 0.0

    @property
    def canary_observation_seconds(self) -> int:
        return int(self._canary.get("observation_seconds", 300))

    @property
    def canary_thresholds(self) -> dict[str, Any]:
        """Operational limits a canary must clear. Never accuracy."""
        return dict(self._canary.get("thresholds") or {})

    # --- drift and retraining (M13) ---------------------------------------
    @property
    def _monitoring(self) -> dict[str, Any]:
        return dict(self.raw.get("monitoring") or {})

    @property
    def _drift(self) -> dict[str, Any]:
        return dict(self._monitoring.get("drift") or {})

    @property
    def drift_reference_split(self) -> str:
        """Split whose distribution a model is held to have learned."""
        return str(self._drift.get("reference_split", "train"))

    @property
    def drift_feature_set(self) -> str:
        return str(self._drift.get("feature_set", "engineered"))

    @property
    def drift_threshold_psi(self) -> float:
        """PSI above which one feature counts as drifted."""
        return float(self._drift.get("threshold_psi", 0.1))

    @property
    def drift_min_features(self) -> int:
        """How many drifted features make it a population-level event.

        One feature can move for a mundane reason. Requiring several is what
        stops every blip churning the registry.
        """
        return int(self._drift.get("min_drifted_features", 2))

    @property
    def _drift_window(self) -> dict[str, Any]:
        return dict(self._drift.get("window") or {})

    @property
    def drift_window_start(self) -> date:
        return _as_date(self._drift_window["start"])

    @property
    def drift_window_end(self) -> date:
        return _as_date(self._drift_window["end"])

    @property
    def drift_scenario(self) -> str:
        """Named controlled scenario, or ``none`` for the untouched window."""
        return str(self._drift_window.get("scenario", "none"))

    @property
    def label_maturity_end(self) -> date:
        """Last approval date whose 60-month horizon has fully elapsed.

        Retraining may only use rows at or before this date. A label is not
        available because the row exists; it is available because its horizon
        closed before the observation cutoff. See docs/drift.md.
        """
        configured = self._monitoring.get("label_maturity_end")
        if configured is not None:
            return _as_date(configured)
        # Derived rather than assumed: observation_end minus the label horizon.
        months = int(self.raw["label"]["horizon_months"])
        cutoff = self.observation_end
        year = cutoff.year - months // 12
        return date(year, cutoff.month, cutoff.day)

    # --- promotion (M6) ---------------------------------------------------
    @property
    def _promotion(self) -> dict[str, Any]:
        return dict(self.raw.get("promotion") or {})

    @property
    def decision_split(self) -> str:
        """Split a promotion is decided on. Never the held-out test split."""
        return str(self._promotion.get("decision_split", "validation"))

    @property
    def registered_model_name(self) -> str:
        return str(self._promotion.get("registered_model_name", "model"))

    @property
    def production_alias(self) -> str:
        return str(self._promotion.get("production_alias", "production"))

    @property
    def bootstrap_model(self) -> str:
        """Stand-in incumbent when nothing is registered yet."""
        return str(self._promotion.get("bootstrap_model", "baseline"))

    @property
    def gate_config(self) -> dict[str, Any]:
        return dict(self._promotion.get("gates") or {})

    # --- optimisation (M5) -----------------------------------------------
    @property
    def _optimization(self) -> dict[str, Any]:
        return dict(self.raw.get("optimization") or {})

    @property
    def n_trials(self) -> int:
        return int(self._optimization.get("n_trials", 25))

    @property
    def sampler_seed(self) -> int:
        """Seeded explicitly; Optuna's default sampler seeds itself from entropy."""
        return int(self._optimization.get("sampler_seed", self.seed))

    @property
    def search_space(self) -> dict[str, Any]:
        return dict(self._optimization.get("search_space") or {})

    @property
    def objective_metric(self) -> str:
        return str(
            dict(self._optimization.get("objective") or {}).get("metric", "average_precision")
        )

    @property
    def objective_split(self) -> str:
        """Split a search is scored on. Never the test split."""
        return str(dict(self._optimization.get("objective") or {}).get("split", "validation"))

    @property
    def objective_direction(self) -> str:
        return str(dict(self._optimization.get("objective") or {}).get("direction", "maximize"))

    @property
    def optimization_experiment_name(self) -> str:
        return str(
            self._optimization.get("experiment_name", f"{self.experiment_name}-optimization")
        )

    @property
    def observation_end(self) -> date:
        return _as_date(self.raw["data"]["observation_end"])

    @property
    def horizon_months(self) -> int:
        return int(self.raw["label"]["horizon_months"])

    @property
    def min_term_months(self) -> int:
        return int(self.raw["population"]["min_term_months"])

    @property
    def target_column(self) -> str:
        return str(self.raw["label"]["target_column"])

    @property
    def split_date_column(self) -> str:
        return str(self.raw["split"]["date_column"])

    @property
    def review_capacity(self) -> float:
        return float(self.raw["evaluation"]["review_capacity"])

    @property
    def primary_metric(self) -> str:
        return str(self.raw["evaluation"]["primary_metric"])

    def window(self, name: str) -> DateWindow:
        """Return the named time-ordered split window."""
        return DateWindow.from_mapping(self.raw["split"][name])

    @property
    def split_names(self) -> list[str]:
        return [k for k in self.raw["split"] if isinstance(self.raw["split"][k], dict)]

    def fingerprint(self) -> str:
        """Stable hash of the resolved configuration, recorded with every run."""
        payload = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


#: Configuration files merged in order. Later files override earlier ones.
CONFIG_LAYERS: tuple[str, ...] = ("base.yaml", "model.yaml")


def load_config(environment: str = "development", config_dir: Path | None = None) -> Config:
    """Load ``base.yaml`` merged with ``model.yaml`` and the environment overlay."""
    directory = config_dir or CONFIG_DIR
    merged: dict[str, Any] = {}
    for name in (*CONFIG_LAYERS, f"{environment}.yaml"):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"missing configuration file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            merged = _deep_merge(merged, yaml.safe_load(handle) or {})
    return Config(raw=merged, environment=environment)
