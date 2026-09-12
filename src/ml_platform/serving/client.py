"""Client for the KServe model tier.

The application tier uses this instead of holding a pipeline of its own when a
predictor URL is configured. It is the other half of
:mod:`ml_platform.serving.predictor`, and it uses the same wire-format functions,
so the two cannot disagree about what a record looks like.

It carries no fallback. If the predictor cannot be reached, the failure is
raised: quietly scoring in process would mean two different things could answer
the same request, and nobody could tell afterwards which one did.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ml_platform.observability import API_SERVICE
from ml_platform.observability.metrics import (
    OUTCOME_ERROR,
    OUTCOME_INVALID,
    OUTCOME_SUCCESS,
    OUTCOME_UNAVAILABLE,
    observe_model_tier,
)
from ml_platform.serving.scoring import frame_to_instances

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0


class PredictorError(RuntimeError):
    """Raised when the model tier cannot be reached or refuses the request."""


class RemotePredictor:
    """Scores by calling a KServe InferenceService."""

    def __init__(
        self,
        url: str,
        model_name: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        #: Which service records the model-tier metrics. The caller is the
        #: application tier, so its own name labels them.
        self.metrics_service = API_SERVICE
        self._client: Any | None = None

    def _http(self) -> Any:
        """One client for the process, so the connection is reused.

        A new connection per request would add a TCP and HTTP handshake to every
        score, which is a meaningful share of the latency for a request this
        small. Created on first use rather than at import.
        """
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def predict_url(self) -> str:
        """KServe's V1 predict path for this model."""
        return f"{self.url}/v1/models/{self.model_name}:predict"

    @property
    def ready_url(self) -> str:
        return f"{self.url}/v1/models/{self.model_name}"

    def ready(self) -> tuple[bool, str | None]:
        """Whether the model tier reports itself able to score."""
        try:
            response = self._http().get(self.ready_url)
        except Exception as exc:
            return False, f"could not reach the predictor at {self.url}: {exc}"
        if response.status_code != 200:
            return False, f"the predictor answered {response.status_code}: {response.text[:200]}"
        payload = response.json()
        if not payload.get("ready"):
            return False, f"the predictor reports {self.model_name!r} as not ready"
        return True, None

    def predict(self, frame: pd.DataFrame) -> list[float]:
        """Probabilities for a register-shaped frame, from the model tier.

        Timed and counted from this side deliberately. The model tier measures
        its own latency too, but only the caller can see queueing, connection
        setup and the network between them -- and a timeout is invisible to the
        service that never answered.
        """
        instances = frame_to_instances(frame)
        started = time.perf_counter()

        try:
            response = self._http().post(self.predict_url, json={"instances": instances})
        except Exception as exc:
            observe_model_tier(
                self.metrics_service, OUTCOME_UNAVAILABLE, time.perf_counter() - started
            )
            raise PredictorError(f"could not reach the predictor at {self.url}: {exc}") from exc

        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            observe_model_tier(self.metrics_service, OUTCOME_ERROR, elapsed)
            raise PredictorError(
                f"the predictor answered {response.status_code}: {response.text[:300]}"
            )
        predictions = response.json().get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(instances):
            observe_model_tier(self.metrics_service, OUTCOME_INVALID, elapsed)
            raise PredictorError(
                f"the predictor returned {predictions!r} for {len(instances)} instance(s)"
            )
        observe_model_tier(self.metrics_service, OUTCOME_SUCCESS, elapsed)
        return [float(p) for p in predictions]


def build_predictor(config: Any) -> RemotePredictor | None:
    """A predictor client when one is configured, otherwise ``None``.

    ``None`` means the application tier loads the model itself, which is how the
    local development server and the plain container run.
    """
    url = config.predictor_url
    if not url:
        return None
    return RemotePredictor(
        url,
        config.predictor_model_name,
        timeout=config.predictor_timeout_seconds,
    )
