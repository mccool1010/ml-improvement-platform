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

from ml_platform.promotion.registry import resolve_production
from ml_platform.serving.canary import TIER_CANARY
from ml_platform.serving.client import RemotePredictor, build_predictor
from ml_platform.serving.scoring import score

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
    #: Set when the model tier is a KServe InferenceService. Then ``pipeline``
    #: is ``None``: this process does not hold the model at all.
    predictor: RemotePredictor | None = None
    #: The canary tier, when one is running. Requests the router sends here are
    #: scored by the candidate instead of the incumbent.
    canary_predictor: RemotePredictor | None = None
    canary_version: str | None = None

    def predict_with(self, tier: str, frame: pd.DataFrame) -> list[float]:
        """Score through a named tier.

        The canary path deliberately has no fallback to production. Quietly
        answering from the incumbent when the candidate fails would hide the
        exact failure the canary exists to surface, and would make the error
        rate the decision rests on read as zero.
        """
        if tier == TIER_CANARY and self.canary_predictor is not None:
            return self.canary_predictor.predict(frame)
        return self.predict(frame)

    @property
    def served_by(self) -> str:
        """Which tier produced the score. Reported, so it is never a guess."""
        return "kserve" if self.predictor is not None else "in-process"

    def predict(self, frame: pd.DataFrame) -> list[float]:
        """Score a frame of applications, returning default probabilities.

        Either the KServe model tier scores it, or this process does. Both run
        :func:`ml_platform.serving.scoring.score` -- the predictor runs it on the
        other side of the wire -- so there is one implementation and the two
        paths cannot answer differently.
        """
        if self.predictor is not None:
            return self.predictor.predict(frame)
        return score(self.pipeline, frame, self.feature_set)


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

    def readiness(self) -> tuple[bool, str | None]:
        """Whether this service can actually serve a prediction right now.

        Two different questions, depending on which tier holds the model.

        In process, it is the startup question and nothing else: the model was
        loaded once, deliberately, so that a request never pays for a registry
        lookup. Nothing can change between probes.

        Against KServe, this process holds no model at all -- only the registry
        metadata it resolved at startup, and a dependency. So readiness asks the
        model tier now. A snapshot taken at startup would leave this service
        reporting ready after the InferenceService had gone away, or reporting
        unready forever because it happened to start first. Checking live is
        what makes the two Deployments independent of their start order.
        """
        if self._model is None:
            return False, self._error or "no production model is resolved"
        predictor = self._model.predictor
        if predictor is None:
            return True, None
        reachable, reason = predictor.ready()
        if not reachable:
            return False, f"the model tier is not ready: {reason}"
        return True, None

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

            tags = self._version_tags(production)

            # Which tier holds the model. The registry lookup above happens
            # either way: the alias decides what production is, and this service
            # reports the version and threshold whichever tier does the scoring.
            predictor = build_predictor(config)
            pipeline = None
            if predictor is None:
                import mlflow.sklearn

                uri = f"models:/{config.registered_model_name}@{config.production_alias}"
                pipeline = mlflow.sklearn.load_model(uri)

            threshold, source = _threshold_from(tags)
            canary_predictor = self._build_canary(config)
            self._model = LoadedModel(
                pipeline=pipeline,
                predictor=predictor,
                canary_predictor=canary_predictor,
                canary_version=self._canary_version(config) if canary_predictor else None,
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
                "resolved %s v%s served %s (feature set %s, threshold %.6f from %s)",
                production.name,
                production.version,
                self._model.served_by,
                self._model.feature_set,
                threshold,
                source,
            )
            return True

        except Exception as exc:
            self._error = f"could not load the production model: {exc}"
            LOGGER.warning(self._error, exc_info=True)
            return False

    def _build_canary(self, config: Config) -> RemotePredictor | None:
        """A client for the canary tier, when the deployment configured one."""
        if not config.canary_enabled:
            return None
        from ml_platform.serving.client import RemotePredictor as Client

        LOGGER.info(
            "canary tier configured at %s (%.2f%% of traffic)",
            config.canary_predictor_url,
            config.canary_traffic_percent,
        )
        return Client(
            str(config.canary_predictor_url),
            config.canary_model_name,
            timeout=config.predictor_timeout_seconds,
        )

    def _canary_version(self, config: Config) -> str | None:
        """Which registered version the canary alias points at, if any."""
        try:
            import mlflow

            mlflow.set_tracking_uri(config.tracking_uri)
            version = mlflow.MlflowClient().get_model_version_by_alias(
                config.registered_model_name, config.canary_alias
            )
            return str(version.version)
        except Exception:
            LOGGER.info("no %r alias is set; the canary version is unknown", config.canary_alias)
            return None

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
