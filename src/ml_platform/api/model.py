"""Model loading for the inference service.

Separate from request handling on purpose. The model is resolved and loaded once
at startup and then reused, so a request never pays for a registry lookup or a
deserialisation.

Resolution goes through the existing registry alias, the same mechanism M6 uses
to decide what production is. The service will not fall back to the newest run,
a file on disk, or anything else: if no model carries the alias, the service
starts and reports itself **not ready** rather than serving something nobody
promoted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ml_platform.features.engineering import build_features
from ml_platform.promotion.registry import resolve_production

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ml_platform.config import Config

LOGGER = logging.getLogger(__name__)

#: Used when the registered version carries no threshold of its own. Recorded in
#: the response so a caller can tell a model's own operating point from this.
FALLBACK_THRESHOLD = 0.5


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before a model is available."""


@dataclass
class LoadedModel:
    """A production model, plus the identity needed to trace a prediction."""

    pipeline: Any
    name: str
    version: str | None
    alias: str
    feature_set: str
    mlflow_run_id: str | None
    platform_run_id: str | None
    decision_threshold: float
    threshold_source: str

    def predict(self, frame: pd.DataFrame) -> list[float]:
        """Score a frame of applications, returning default probabilities.

        Features are built by the same code that built them for training, so the
        served representation cannot drift from the trained one.
        """
        features = build_features(frame, self.feature_set)
        return [float(p) for p in self.pipeline.predict_proba(features)[:, 1]]


class ModelService:
    """Holds the loaded model for the lifetime of the process."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: LoadedModel | None = None
        self._error: str | None = None

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            raise ModelNotLoadedError(self._error or "no production model is loaded")
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> bool:
        """Resolve and load the production model. Returns whether it succeeded.

        Failure is recorded rather than raised: a service that cannot load a
        model should start and report itself unready, so an orchestrator can see
        the reason instead of a crash loop.
        """
        config = self._config
        try:
            production = resolve_production(config)
            if production is None:
                self._error = (
                    f"no model carries the {config.production_alias!r} alias on "
                    f"{config.registered_model_name!r}; promote a candidate first"
                )
                LOGGER.warning(self._error)
                return False

            import mlflow.sklearn

            uri = f"models:/{config.registered_model_name}@{config.production_alias}"
            pipeline = mlflow.sklearn.load_model(uri)
            tags = self._version_tags(production)

            threshold, source = _threshold_from(tags)
            self._model = LoadedModel(
                pipeline=pipeline,
                name=production.name,
                version=production.version,
                alias=config.production_alias,
                feature_set=tags.get("feature_set", "engineered"),
                mlflow_run_id=production.run_id,
                platform_run_id=production.platform_run_id,
                decision_threshold=threshold,
                threshold_source=source,
            )
            self._error = None
            LOGGER.info(
                "loaded %s v%s (feature set %s, threshold %.6f from %s)",
                production.name,
                production.version,
                self._model.feature_set,
                threshold,
                source,
            )
            return True

        except Exception as exc:
            self._error = f"could not load the production model: {exc}"
            LOGGER.warning(self._error, exc_info=True)
            return False

    def _version_tags(self, production: Any) -> dict[str, str]:
        """Registry tags for the resolved version, or an empty mapping."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._config.tracking_uri)
            version = mlflow.MlflowClient().get_model_version(
                production.name, str(production.version)
            )
            return dict(version.tags or {})
        except Exception:
            LOGGER.warning("could not read version tags", exc_info=True)
            return {}


def _threshold_from(tags: dict[str, str]) -> tuple[float, str]:
    """The model's own operating point, if it recorded one.

    A single prediction cannot compute the review-capacity quantile the project
    evaluates at, so the threshold is whatever the model measured at promotion
    time. When a version predates that tag, a neutral fallback is used and the
    response says so, rather than presenting a borrowed number as the model's.
    """
    raw = tags.get("validation_threshold_at_capacity")
    if raw is None:
        return FALLBACK_THRESHOLD, "fallback"
    try:
        return float(raw), "registry"
    except ValueError:
        return FALLBACK_THRESHOLD, "fallback"
