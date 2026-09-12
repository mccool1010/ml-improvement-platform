"""Where canary measurements come from.

Two implementations of the same small interface.

:class:`InProcessSignals` reads the counters this process has been incrementing,
which is what makes a canary decision testable: a test can drive requests through
the real application and evaluate the real signals with no cluster, no scrape and
no waiting.

:class:`PrometheusSignals` queries the M12 Prometheus, which is what makes a
decision *correct* in the cluster: with two API replicas, in-process counters see
only the requests that happened to reach one of them, and a canary judged on half
the evidence is a canary judged wrongly.

Neither is a fallback for the other. The evaluator is handed whichever one suits
where it is running, and the decision report records which was used, because
"5 errors out of 200" means something different depending on whether that was all
the traffic or one replica's share of it.
"""

from __future__ import annotations

import logging
from typing import Any

from ml_platform.monitoring.canary import TierSignals
from ml_platform.serving.canary import TIER_CANARY, TIER_PRODUCTION

LOGGER = logging.getLogger(__name__)


class SignalError(RuntimeError):
    """Raised when measurements cannot be obtained at all."""


class InProcessSignals:
    """Signals from this process's own Prometheus client registry.

    Counters are cumulative, so a window is expressed as the difference from a
    baseline taken when the canary started. Without that, an evaluation would
    include every request since the process booted.
    """

    def __init__(self, registry: Any | None = None) -> None:
        from prometheus_client import REGISTRY

        self._registry = registry or REGISTRY
        self._baseline: dict[str, dict[str, float]] = {}

    def _sample(self, name: str, labels: dict[str, str]) -> float:
        value = self._registry.get_sample_value(name, labels)
        return 0.0 if value is None else float(value)

    def _raw(self, tier: str) -> dict[str, float]:
        """Cumulative counters for one tier, right now."""
        requests = self._sample("ml_platform_canary_requests_total", {"tier": tier})
        errors = self._sample("ml_platform_canary_request_errors_total", {"tier": tier})
        failures = self._sample("ml_platform_canary_upstream_failures_total", {"tier": tier})
        duration_sum = self._sample(
            "ml_platform_canary_request_duration_seconds_sum", {"tier": tier}
        )
        duration_count = self._sample(
            "ml_platform_canary_request_duration_seconds_count", {"tier": tier}
        )
        return {
            "requests": requests,
            "errors": errors,
            "upstream_failures": failures,
            "duration_sum": duration_sum,
            "duration_count": duration_count,
        }

    def _quantile(self, tier: str, quantile: float) -> float:
        """p95 from the histogram buckets, using only the window's increments."""
        from ml_platform.observability.metrics import LATENCY_BUCKETS

        edges = [*LATENCY_BUCKETS, float("inf")]
        counts: list[float] = []
        for edge in edges:
            label = "+Inf" if edge == float("inf") else _format_bucket(edge)
            cumulative = self._sample(
                "ml_platform_canary_request_duration_seconds_bucket",
                {"tier": tier, "le": label},
            )
            base = self._baseline.get(f"{tier}:bucket:{label}", {}).get("value", 0.0)
            counts.append(max(cumulative - base, 0.0))

        total = counts[-1]
        if total <= 0:
            return 0.0
        target = quantile * total
        for edge, cumulative in zip(edges, counts, strict=True):
            if cumulative >= target:
                # The bucket's upper edge, which is the conventional
                # conservative reading of a histogram quantile.
                return float(edge) if edge != float("inf") else float(LATENCY_BUCKETS[-1])
        return float(LATENCY_BUCKETS[-1])

    def mark_window_start(self) -> None:
        """Record the baseline a later evaluation is measured against."""
        from ml_platform.observability.metrics import LATENCY_BUCKETS

        self._baseline = {}
        for tier in (TIER_CANARY, TIER_PRODUCTION):
            self._baseline[tier] = self._raw(tier)
            for edge in [*LATENCY_BUCKETS, float("inf")]:
                label = "+Inf" if edge == float("inf") else _format_bucket(edge)
                self._baseline[f"{tier}:bucket:{label}"] = {
                    "value": self._sample(
                        "ml_platform_canary_request_duration_seconds_bucket",
                        {"tier": tier, "le": label},
                    )
                }

    def signals_for(self, tier: str, window_seconds: int) -> TierSignals:
        del window_seconds  # the window is the baseline, not a duration
        now = self._raw(tier)
        base = self._baseline.get(tier, {})

        def _delta(key: str) -> float:
            return max(now[key] - base.get(key, 0.0), 0.0)

        requests = int(_delta("requests"))
        count = _delta("duration_count")
        mean = _delta("duration_sum") / count if count else 0.0
        healthy = self._sample("ml_platform_canary_tier_healthy", {"tier": tier})

        return TierSignals(
            tier=tier,
            requests=requests,
            errors=int(_delta("errors")),
            upstream_failures=int(_delta("upstream_failures")),
            latency_p95_seconds=self._quantile(tier, 0.95),
            latency_mean_seconds=mean,
            healthy=healthy >= 1.0,
            health_detail=None if healthy >= 1.0 else "the tier reported itself unhealthy",
        )


def _format_bucket(edge: float) -> str:
    """Bucket label as prometheus_client writes it."""
    return repr(float(edge))


class PrometheusSignals:
    """Signals from the M12 Prometheus, aggregated across every replica."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _query(self, expression: str) -> float:
        import httpx

        try:
            response = httpx.get(
                f"{self.base_url}/api/v1/query",
                params={"query": expression},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()["data"]["result"]
        except Exception as exc:
            raise SignalError(f"could not query Prometheus at {self.base_url}: {exc}") from exc

        if not result:
            return 0.0
        value = float(result[0]["value"][1])
        # A quantile over an empty window is NaN, which must not be mistaken for
        # zero latency -- that would silently pass a latency check.
        return 0.0 if value != value else value

    def signals_for(self, tier: str, window_seconds: int) -> TierSignals:
        window = f"{window_seconds}s"
        selector = f'{{tier="{tier}"}}'
        requests = self._query(
            f"sum(increase(ml_platform_canary_requests_total{selector}[{window}]))"
        )
        errors = self._query(
            f"sum(increase(ml_platform_canary_request_errors_total{selector}[{window}]))"
        )
        failures = self._query(
            f"sum(increase(ml_platform_canary_upstream_failures_total{selector}[{window}]))"
        )
        p95 = self._query(
            "histogram_quantile(0.95, sum by (le) (rate("
            f"ml_platform_canary_request_duration_seconds_bucket{selector}[{window}])))"
        )
        total = self._query(
            f"sum(increase(ml_platform_canary_request_duration_seconds_sum{selector}[{window}]))"
        )
        count = self._query(
            f"sum(increase(ml_platform_canary_request_duration_seconds_count{selector}[{window}]))"
        )
        healthy = self._query(f"min(ml_platform_canary_tier_healthy{selector})")

        return TierSignals(
            tier=tier,
            requests=round(requests),
            errors=round(errors),
            upstream_failures=round(failures),
            latency_p95_seconds=p95,
            latency_mean_seconds=total / count if count else 0.0,
            healthy=healthy >= 1.0,
            health_detail=None if healthy >= 1.0 else "the tier reported itself unhealthy",
        )
