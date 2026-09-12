"""Splitting traffic between the incumbent and a canary.

KServe runs here in RawDeployment mode, which creates a plain Deployment and
Service per InferenceService and has **no traffic-splitting primitive**;
``canaryTrafficPercent`` is a Serverless-mode feature, and that gap was recorded
as debt at M11. The three ways to close it are Knative, a service mesh, or
splitting in the tier that already fronts the model. ADR-002 already makes
FastAPI that tier, so the split happens here: no new infrastructure, no mesh, and
KServe still serves each model version through its own InferenceService. The
canary is a second InferenceService, not a second way of serving.

**Routing is deterministic, not random.** A percentage implemented with
``random()`` cannot be tested, cannot be reproduced from a report, and sends the
same caller to different models on consecutive identical requests. Instead a
routing key is hashed into 10,000 buckets and compared against the allocation.
The consequences are worth having:

* the same application always reaches the same tier, so a caller who retries
  gets a consistent answer;
* a test asserts an exact split rather than a statistical one;
* a decision report can name the allocation and anyone can reproduce which
  requests it covered.

The router holds no opinion about whether the canary is any good. It moves
traffic; :mod:`ml_platform.monitoring.canary` decides.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.serving.client import RemotePredictor

LOGGER = logging.getLogger(__name__)

#: Buckets the routing key is hashed into. Ten thousand gives a resolution of
#: 0.01%, which is finer than any allocation this project would choose and keeps
#: the arithmetic in integers.
BUCKETS = 10_000

#: Header a caller may set to pin its own routing. Useful for a smoke test that
#: wants to reach a specific tier, and for a client that wants stable answers
#: across a session.
ROUTING_KEY_HEADER = "x-ml-routing-key"

#: The two tiers, as they appear in metrics and reports.
TIER_PRODUCTION = "production"
TIER_CANARY = "canary"


#: Highest allocation a canary may hold. Strictly below 100: at 100 the
#: incumbent receives nothing, which is a full cutover to an unproven model
#: wearing a canary's name. Whatever that is, it is not a canary, and the point
#: of the arrangement is that the incumbent stays there to fall back to.
MAX_TRAFFIC_PERCENT = 99.0


class CanaryError(RuntimeError):
    """Raised when a canary configuration cannot be honoured."""


def _validate_traffic(percent: float) -> None:
    """Allocations must leave the incumbent serving something."""
    if not 0.0 <= percent <= MAX_TRAFFIC_PERCENT:
        raise CanaryError(
            f"canary traffic must be between 0 and {MAX_TRAFFIC_PERCENT} percent, got {percent}. "
            "An allocation of 100 leaves the incumbent with no traffic, so there is nothing "
            "to compare against and nothing already serving to fall back to."
        )


def routing_bucket(key: str) -> int:
    """Stable bucket in ``[0, BUCKETS)`` for a routing key.

    SHA-256 rather than :func:`hash`, which is randomised per process and would
    send the same key to different tiers in different replicas.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def routing_key_for(payload: Any, header_value: str | None = None) -> str:
    """The key a request routes on.

    An explicit header wins. Otherwise the key is a digest of the request
    content, which makes routing sticky per application: the same loan reaches
    the same tier every time it is scored.
    """
    if header_value:
        return header_value
    try:
        rendered = json.dumps(payload, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        rendered = repr(payload)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass
class CanaryState:
    """What the canary is, and how much traffic it is allowed.

    Deliberately small and inspectable: everything a decision report needs to
    say which model saw which share of traffic.
    """

    enabled: bool = False
    traffic_percent: float = 0.0
    candidate_version: str | None = None
    incumbent_version: str | None = None
    candidate_url: str | None = None
    #: Set when the canary has been stopped, so a report can say why.
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_traffic(self.traffic_percent)

    @property
    def threshold(self) -> int:
        """Bucket below which a request goes to the canary."""
        return round(self.traffic_percent / 100.0 * BUCKETS)

    @property
    def active(self) -> bool:
        """Whether any traffic should actually reach the canary."""
        return self.enabled and self.traffic_percent > 0.0 and self.candidate_url is not None

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": self.active,
            "traffic_percent": self.traffic_percent,
            "candidate_version": self.candidate_version,
            "incumbent_version": self.incumbent_version,
            "candidate_url": self.candidate_url,
            "stopped_reason": self.stopped_reason,
        }


class CanaryRouter:
    """Chooses a tier for each request, and can be stopped instantly."""

    def __init__(self, state: CanaryState | None = None) -> None:
        self._state = state or CanaryState()

    @property
    def state(self) -> CanaryState:
        return self._state

    @property
    def traffic_percent(self) -> float:
        return self._state.traffic_percent if self._state.active else 0.0

    def tier_for(self, key: str) -> str:
        """Which tier this routing key belongs to."""
        if not self._state.active:
            return TIER_PRODUCTION
        return TIER_CANARY if routing_bucket(key) < self._state.threshold else TIER_PRODUCTION

    def set_traffic(self, percent: float) -> None:
        """Change the allocation. Used to dial up, and to roll back to zero."""
        _validate_traffic(percent)
        self._state.traffic_percent = percent

    def stop(self, reason: str) -> None:
        """Send every request back to the incumbent, immediately.

        This is the rollback. It is one assignment rather than a redeploy,
        because the thing a rollback must be is *fast* -- the next request after
        this call reaches production, with no pod restart, no registry write and
        no traffic draining to wait for.
        """
        self._state.enabled = False
        self._state.traffic_percent = 0.0
        self._state.stopped_reason = reason
        LOGGER.warning("canary stopped: %s", reason)

    def start(
        self,
        *,
        traffic_percent: float,
        candidate_url: str,
        candidate_version: str | None = None,
        incumbent_version: str | None = None,
    ) -> None:
        """Begin sending a share of traffic to the candidate."""
        self._state = CanaryState(
            enabled=True,
            traffic_percent=traffic_percent,
            candidate_version=candidate_version,
            incumbent_version=incumbent_version,
            candidate_url=candidate_url,
        )
        LOGGER.info(
            "canary started: %.2f%% to version %s at %s",
            traffic_percent,
            candidate_version,
            candidate_url,
        )

    def observed_split(self, keys: list[str]) -> dict[str, int]:
        """How a set of keys would actually divide. Used by tests and reports.

        The realised split is reported rather than the configured one, because
        hashing a finite number of keys never lands exactly on the percentage
        and a report that claimed otherwise would be wrong.
        """
        counts = {TIER_PRODUCTION: 0, TIER_CANARY: 0}
        for key in keys:
            counts[self.tier_for(key)] += 1
        return counts


def build_canary_predictor(config: Any, state: CanaryState) -> RemotePredictor | None:
    """A client for the canary InferenceService, when one is configured."""
    if not state.candidate_url:
        return None
    from ml_platform.serving.client import RemotePredictor

    return RemotePredictor(
        state.candidate_url,
        config.canary_model_name,
        timeout=config.predictor_timeout_seconds,
    )
