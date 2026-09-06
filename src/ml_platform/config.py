"""Configuration loading.

Configuration is layered: ``base.yaml`` always applies, and an environment file
(``development``/``production``) overrides it. Every run records the resolved
configuration and its hash so a result can be tied back to the exact settings
that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ml_platform.paths import project_root, resolve

CONFIG_DIR = project_root() / "configs"


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
        """
        raw = str(self._tracking.get("backend_uri", "sqlite:///mlflow.db"))
        prefix = "sqlite:///"
        if raw.startswith(prefix):
            target = raw[len(prefix) :]
            if not Path(target).is_absolute():
                return prefix + resolve(target).as_posix()
        return raw

    @property
    def artifact_uri(self) -> str:
        """Where MLflow stores run artifacts, as a file URI under the project."""
        return resolve(str(self._tracking.get("artifact_dir", "mlartifacts"))).as_uri()

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
