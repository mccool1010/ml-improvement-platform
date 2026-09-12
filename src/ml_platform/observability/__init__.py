"""Telemetry for both serving tiers.

:mod:`ml_platform.observability.metrics` is what Prometheus scrapes;
:mod:`ml_platform.observability.tracing` is what Jaeger receives. Both are
installed by the application factories and neither may raise into a request.

The division of labour is the usual one and worth stating, because it decides
what belongs where. Metrics answer "how much, how often, how slow" across all
traffic and are cheap enough to keep forever. Traces answer "what happened to
*this* request, and where did the time go" across service boundaries. Anything
per-request goes in a span, never in a metric label.
"""

from ml_platform.observability import metrics, tracing

#: Metric and span label for the application tier.
API_SERVICE = "inference-api"
#: Metric and span label for the KServe model tier.
PREDICTOR_SERVICE = "model-predictor"

__all__ = ["API_SERVICE", "PREDICTOR_SERVICE", "metrics", "tracing"]
