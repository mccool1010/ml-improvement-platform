"""Deciding whether a canary may become production.

Every signal here is **operational**, observable within minutes, and available
without a single label: HTTP error rate, latency, upstream failure rate, and
whether the candidate can serve at all. That restriction is not a simplification,
it is ADR-003's rule. Realised performance on this dataset takes five years to
arrive, so a canary that waited for accuracy would never conclude, and a canary
that used an *estimate* of accuracy would be acting on a number nobody can
check. Model quality was already judged offline by the M6 gates before any
traffic moved; what a canary adds is the question those gates cannot answer --
does this model behave itself when real requests hit it.

So the decision this module makes is narrow and worth stating plainly: **the
candidate is not visibly worse to operate than the incumbent.** It is not "the
candidate is better". A canary that passes has earned the alias only in the sense
that nothing went wrong.

Two comparisons are made for each signal, and both matter:

*Against an absolute ceiling*, because an error rate of 40% is unacceptable
however badly the incumbent is doing.

*Against the incumbent*, because a candidate is being asked to replace something
specific. An absolute-only check passes a candidate that is twice as slow as
production but still inside the ceiling.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ml_platform.serving.canary import TIER_CANARY, TIER_PRODUCTION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.config import Config

LOGGER = logging.getLogger(__name__)

#: Decisions a canary evaluation can reach. `hold` is not a failure: it means
#: not enough traffic has arrived to say anything, and deciding anyway would be
#: reading noise.
DECISION_PROMOTE = "promote"
DECISION_ROLLBACK = "rollback"
DECISION_HOLD = "hold"


@dataclass
class TierSignals:
    """What one tier did during the observation window."""

    tier: str
    requests: int = 0
    errors: int = 0
    upstream_failures: int = 0
    latency_p95_seconds: float = 0.0
    latency_mean_seconds: float = 0.0
    healthy: bool = True
    health_detail: str | None = None

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    @property
    def failure_rate(self) -> float:
        """Upstream timeouts and unreachable model tier, as a share of requests."""
        return self.upstream_failures / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "error_rate": round(self.error_rate, 6),
            "failure_rate": round(self.failure_rate, 6),
        }


class SignalSource(Protocol):
    """Where measurements come from.

    An interface rather than a direct Prometheus call, so the decision logic can
    be tested exactly -- a canary evaluator that can only be exercised against a
    live cluster is one nobody will exercise.
    """

    def signals_for(self, tier: str, window_seconds: int) -> TierSignals: ...


@dataclass
class CanaryCheck:
    """One threshold, and whether the candidate cleared it."""

    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    detail: str
    blocking: bool = True

    def describe(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name:<26} observed {self.observed}, allowed {self.threshold}"


@dataclass
class CanaryDecision:
    """The verdict, and everything needed to justify it afterwards."""

    canary_event_id: str
    decision: str
    reason: str
    candidate_version: str | None
    incumbent_version: str | None
    traffic_percent: float
    observation_window_seconds: int
    canary: TierSignals
    production: TierSignals
    checks: list[CanaryCheck] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[CanaryCheck]:
        return [c for c in self.checks if not c.passed and c.blocking]

    @property
    def should_rollback(self) -> bool:
        return self.decision == DECISION_ROLLBACK

    @property
    def should_promote(self) -> bool:
        return self.decision == DECISION_PROMOTE

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"[{self.decision.upper()}] {passed}/{len(self.checks)} checks passed "
            f"at {self.traffic_percent}% traffic over {self.observation_window_seconds}s: "
            f"{self.reason}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canary_event_id": self.canary_event_id,
            "decision": self.decision,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
            "candidate_version": self.candidate_version,
            "incumbent_version": self.incumbent_version,
            "traffic_percent": self.traffic_percent,
            "observation_window_seconds": self.observation_window_seconds,
            "thresholds": self.thresholds,
            "signals": {
                TIER_CANARY: self.canary.to_dict(),
                TIER_PRODUCTION: self.production.to_dict(),
            },
            "checks": [asdict(c) for c in self.checks],
            "failed_checks": [c.name for c in self.failures],
            "metadata": self.metadata,
        }


def _ratio(candidate: float, incumbent: float) -> float:
    """Candidate relative to incumbent, with a sane answer at zero.

    A zero incumbent is the interesting edge: any candidate failure is then
    infinitely worse in relative terms, so the absolute ceiling is what governs
    and this returns 1.0 rather than an infinity that would fail every time.
    """
    if incumbent <= 0:
        return 1.0 if candidate <= 0 else float("inf")
    return candidate / incumbent


def evaluate_canary(
    *,
    canary: TierSignals,
    production: TierSignals,
    thresholds: dict[str, Any],
    canary_event_id: str,
    traffic_percent: float,
    observation_window_seconds: int,
    candidate_version: str | None = None,
    incumbent_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CanaryDecision:
    """Judge a canary on operational evidence. Pure: no I/O, no side effects."""
    checks: list[CanaryCheck] = []

    min_requests = int(thresholds.get("min_requests", 50))
    max_error_rate = float(thresholds.get("max_error_rate", 0.02))
    max_error_rate_ratio = float(thresholds.get("max_error_rate_ratio", 2.0))
    max_failure_rate = float(thresholds.get("max_failure_rate", 0.01))
    max_latency_p95 = float(thresholds.get("max_latency_p95_seconds", 1.0))
    max_latency_ratio = float(thresholds.get("max_latency_ratio", 1.5))

    # Health first. An unhealthy candidate is decided immediately: there is no
    # point measuring the latency of a tier that cannot serve.
    checks.append(
        CanaryCheck(
            name="candidate_healthy",
            passed=canary.healthy,
            observed="healthy" if canary.healthy else "unhealthy",
            threshold="healthy",
            detail=canary.health_detail or "the candidate reports itself able to serve",
        )
    )
    if not canary.healthy:
        return CanaryDecision(
            canary_event_id=canary_event_id,
            decision=DECISION_ROLLBACK,
            reason=f"the candidate is not serving: {canary.health_detail or 'unhealthy'}",
            candidate_version=candidate_version,
            incumbent_version=incumbent_version,
            traffic_percent=traffic_percent,
            observation_window_seconds=observation_window_seconds,
            canary=canary,
            production=production,
            checks=checks,
            thresholds=dict(thresholds),
            metadata=metadata or {},
        )

    # Enough traffic to say anything. Non-blocking: too little evidence is a
    # reason to wait, never a reason to roll back a candidate that has done
    # nothing wrong.
    enough = canary.requests >= min_requests
    checks.append(
        CanaryCheck(
            name="sufficient_traffic",
            passed=enough,
            observed=canary.requests,
            threshold=min_requests,
            detail="deciding on fewer requests than this would be reading noise",
            blocking=False,
        )
    )

    checks.append(
        CanaryCheck(
            name="error_rate_absolute",
            passed=canary.error_rate <= max_error_rate,
            observed=round(canary.error_rate, 6),
            threshold=max_error_rate,
            detail="share of canary requests answered 4xx or 5xx",
        )
    )
    error_ratio = _ratio(canary.error_rate, production.error_rate)
    checks.append(
        CanaryCheck(
            name="error_rate_vs_incumbent",
            passed=error_ratio <= max_error_rate_ratio,
            observed=round(error_ratio, 4),
            threshold=max_error_rate_ratio,
            detail="canary error rate as a multiple of the incumbent's",
        )
    )
    checks.append(
        CanaryCheck(
            name="upstream_failure_rate",
            passed=canary.failure_rate <= max_failure_rate,
            observed=round(canary.failure_rate, 6),
            threshold=max_failure_rate,
            detail="timeouts and unreachable model tier, which a 200 rate hides",
        )
    )
    checks.append(
        CanaryCheck(
            name="latency_p95_absolute",
            passed=canary.latency_p95_seconds <= max_latency_p95,
            observed=round(canary.latency_p95_seconds, 6),
            threshold=max_latency_p95,
            detail="p95 latency of canary requests",
        )
    )
    latency_ratio = _ratio(canary.latency_p95_seconds, production.latency_p95_seconds)
    checks.append(
        CanaryCheck(
            name="latency_vs_incumbent",
            passed=latency_ratio <= max_latency_ratio,
            observed=round(latency_ratio, 4),
            threshold=max_latency_ratio,
            detail="canary p95 as a multiple of the incumbent's",
        )
    )

    failures = [c for c in checks if not c.passed and c.blocking]
    if failures:
        decision, reason = (
            DECISION_ROLLBACK,
            "; ".join(f"{c.name} ({c.observed} > {c.threshold})" for c in failures),
        )
    elif not enough:
        decision, reason = (
            DECISION_HOLD,
            f"only {canary.requests} canary request(s); {min_requests} needed to decide",
        )
    else:
        decision, reason = (
            DECISION_PROMOTE,
            "the candidate is not measurably worse to operate than the incumbent",
        )

    result = CanaryDecision(
        canary_event_id=canary_event_id,
        decision=decision,
        reason=reason,
        candidate_version=candidate_version,
        incumbent_version=incumbent_version,
        traffic_percent=traffic_percent,
        observation_window_seconds=observation_window_seconds,
        canary=canary,
        production=production,
        checks=checks,
        thresholds=dict(thresholds),
        metadata=metadata or {},
    )
    LOGGER.info(result.summary())
    return result


def thresholds_from(config: Config) -> dict[str, Any]:
    """The configured canary thresholds."""
    return dict(config.canary_thresholds)
