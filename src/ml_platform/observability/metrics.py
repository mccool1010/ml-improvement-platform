"""Prometheus metrics for both tiers.

What is measured here is what an operator can act on: how much traffic arrived,
how much of it failed, how long it took, and whether the model tier answered.
That is also the list ADR-003 draws rollback authority from -- error rate,
latency, timeouts and serving health are operational signals, and operational
signals may act immediately. Accuracy is not here and must not be: realised
performance takes years to arrive and belongs to a different clock.

**Cardinality is the thing to get wrong.** A Prometheus series is created per
distinct label combination and lives in memory for as long as it is retained, so
a label carrying a request id, a customer, an amount, or a raw URL path turns one
metric into unbounded many and eventually takes the server down. Every label used
here is drawn from a small fixed set: the route *template* rather than the path,
the HTTP method, the status code, and a handful of named outcomes. The one label
with any real breadth is the served model version, which is deliberate -- being
able to attribute latency or errors to a specific promoted version is the point
of the whole registry -- and it is bounded by how often a model is promoted.

Nothing in this module may raise into a request. A platform whose telemetry can
break the thing it observes is worse than one with no telemetry at all.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI, Request, Response

LOGGER = logging.getLogger(__name__)

#: Route label for a request that matched no route. The raw path of a 404 is
#: attacker-controlled and unbounded, so it is never used as a label value.
UNMATCHED_ROUTE = "unmatched"

#: Buckets in seconds. Chosen for this service: a local score is around a
#: millisecond, a call to the model tier is a few milliseconds, and anything past
#: a second means something is wrong. The default buckets are too coarse at the
#: fast end to show any of that.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

# --- HTTP ------------------------------------------------------------------

REQUESTS = Counter(
    "ml_platform_http_requests_total",
    "HTTP requests, by route template and status code.",
    ["service", "method", "route", "status"],
)

REQUEST_ERRORS = Counter(
    "ml_platform_http_request_errors_total",
    "HTTP requests that ended in a 4xx or 5xx, by class.",
    ["service", "method", "route", "status_class"],
)

REQUEST_DURATION = Histogram(
    "ml_platform_http_request_duration_seconds",
    "Wall-clock time to produce a response.",
    ["service", "method", "route"],
    buckets=LATENCY_BUCKETS,
)

# --- predictions -----------------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "ml_platform_prediction_requests_total",
    "Calls to /predict, by outcome.",
    ["service", "outcome"],
)

APPLICATIONS_SCORED = Counter(
    "ml_platform_applications_scored_total",
    "Individual loan applications scored. A request may carry many.",
    ["service"],
)

APPLICATIONS_FLAGGED = Counter(
    "ml_platform_applications_flagged_total",
    "Applications whose probability reached the model's decision threshold.",
    ["service"],
)

# --- the model tier --------------------------------------------------------

MODEL_TIER_REQUESTS = Counter(
    "ml_platform_model_tier_requests_total",
    "Calls from the application tier to the KServe model tier, by outcome.",
    ["service", "outcome"],
)

MODEL_TIER_DURATION = Histogram(
    "ml_platform_model_tier_duration_seconds",
    "Time spent waiting on the model tier, measured by the caller.",
    ["service"],
    buckets=LATENCY_BUCKETS,
)

MODEL_READY = Gauge(
    "ml_platform_model_ready",
    "1 when this service can serve a prediction right now, 0 otherwise.",
    ["service"],
)

MODEL_INFO = Gauge(
    "ml_platform_model_info",
    "Always 1. The labels carry which model version is being served.",
    ["service", "model_name", "version", "alias", "served_by"],
)

# --- canary (M14) ----------------------------------------------------------
#
# Separate series rather than a `tier` label on the metrics above. Adding a label
# to an existing metric changes every series it produces, which would break the
# M12 dashboard queries and the alerts written against them. These are additive:
# nothing in M12 sees any difference.
#
# `tier` takes exactly two values, production and canary.

CANARY_REQUESTS = Counter(
    "ml_platform_canary_requests_total",
    "Requests routed to each tier while a canary is running.",
    ["tier"],
)

CANARY_REQUEST_ERRORS = Counter(
    "ml_platform_canary_request_errors_total",
    "Requests to each tier that ended in a 4xx or 5xx.",
    ["tier"],
)

CANARY_UPSTREAM_FAILURES = Counter(
    "ml_platform_canary_upstream_failures_total",
    "Timeouts and unreachable model tiers, which a status-code count hides.",
    ["tier"],
)

CANARY_REQUEST_DURATION = Histogram(
    "ml_platform_canary_request_duration_seconds",
    "Time to score a request, by the tier that served it.",
    ["tier"],
    buckets=LATENCY_BUCKETS,
)

CANARY_TIER_HEALTHY = Gauge(
    "ml_platform_canary_tier_healthy",
    "1 when the tier reports itself able to serve, 0 otherwise.",
    ["tier"],
)

CANARY_TRAFFIC_PERCENT = Gauge(
    "ml_platform_canary_traffic_percent",
    "Share of traffic currently allocated to the canary. 0 when none is running.",
)


def observe_canary_request(
    tier: str,
    *,
    duration_seconds: float,
    failed: bool = False,
    upstream_failure: bool = False,
) -> None:
    """Record one request against the tier that served it."""
    try:
        CANARY_REQUESTS.labels(tier).inc()
        CANARY_REQUEST_DURATION.labels(tier).observe(duration_seconds)
        if failed:
            CANARY_REQUEST_ERRORS.labels(tier).inc()
        if upstream_failure:
            CANARY_UPSTREAM_FAILURES.labels(tier).inc()
    except Exception:  # pragma: no cover - telemetry must never break a request
        LOGGER.debug("failed to record canary metrics", exc_info=True)


def set_canary_tier_healthy(tier: str, healthy: bool) -> None:
    try:
        CANARY_TIER_HEALTHY.labels(tier).set(1 if healthy else 0)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to set the canary health gauge", exc_info=True)


def set_canary_traffic(percent: float) -> None:
    """Publish the current allocation, so a dashboard shows a rollback landing."""
    try:
        CANARY_TRAFFIC_PERCENT.set(percent)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to set the canary traffic gauge", exc_info=True)


#: Outcomes for the model tier. A fixed vocabulary, so the label stays bounded
#: and a dashboard can name each case.
OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_INVALID = "invalid"


def _status_class(status_code: int) -> str:
    """``2xx``, ``4xx`` and so on. Four values, not six hundred."""
    return f"{status_code // 100}xx"


def observe_request(
    service: str, method: str, route: str, status_code: int, duration_seconds: float
) -> None:
    """Record one finished HTTP request."""
    try:
        status = str(status_code)
        REQUESTS.labels(service, method, route, status).inc()
        REQUEST_DURATION.labels(service, method, route).observe(duration_seconds)
        if status_code >= 400:
            REQUEST_ERRORS.labels(service, method, route, _status_class(status_code)).inc()
    except Exception:  # pragma: no cover - metrics must never break a request
        LOGGER.debug("failed to record request metrics", exc_info=True)


def observe_prediction(service: str, outcome: str, scored: int = 0, flagged: int = 0) -> None:
    """Record one call to /predict and what it produced."""
    try:
        PREDICTION_REQUESTS.labels(service, outcome).inc()
        if scored:
            APPLICATIONS_SCORED.labels(service).inc(scored)
        if flagged:
            APPLICATIONS_FLAGGED.labels(service).inc(flagged)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to record prediction metrics", exc_info=True)


def observe_model_tier(service: str, outcome: str, duration_seconds: float | None = None) -> None:
    """Record one call to the KServe model tier, from the caller's side."""
    try:
        MODEL_TIER_REQUESTS.labels(service, outcome).inc()
        if duration_seconds is not None:
            MODEL_TIER_DURATION.labels(service).observe(duration_seconds)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to record model tier metrics", exc_info=True)


def set_model_ready(service: str, ready: bool) -> None:
    try:
        MODEL_READY.labels(service).set(1 if ready else 0)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to set readiness gauge", exc_info=True)


def set_model_info(
    service: str, model_name: str, version: str | None, alias: str, served_by: str
) -> None:
    """Publish which version is being served.

    Cleared first: leaving the previous version's series behind would make a
    dashboard show two production models at once after a promotion.
    """
    try:
        MODEL_INFO.clear()
        MODEL_INFO.labels(service, model_name, version or "unknown", alias, served_by).set(1)
    except Exception:  # pragma: no cover
        LOGGER.debug("failed to set model info gauge", exc_info=True)


def route_template(request: Any) -> str:
    """The matched route's template, never the raw path.

    ``/predict`` is one series. The raw path of an unmatched request is whatever
    the caller sent, so it collapses to a single constant instead.
    """
    try:
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            return path
    except Exception:  # pragma: no cover
        pass
    return UNMATCHED_ROUTE


def install(app: FastAPI, service: str) -> None:
    """Add the request middleware and the ``/metrics`` endpoint to an app.

    The middleware records after the handler returns, so the route has been
    matched by then and the template is available.
    """
    from fastapi import Response

    @app.middleware("http")
    async def _record(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # An unhandled exception is still a 500 from the caller's point of
            # view, and is the most important thing on the error graph.
            observe_request(
                service,
                request.method,
                route_template(request),
                500,
                time.perf_counter() - started,
            )
            raise
        observe_request(
            service,
            request.method,
            route_template(request),
            response.status_code,
            time.perf_counter() - started,
        )
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
