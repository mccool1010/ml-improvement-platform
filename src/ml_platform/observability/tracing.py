"""OpenTelemetry tracing, exported to Jaeger over OTLP.

A prediction crosses two services: the application tier validates it and the
model tier scores it. When one of them is slow, a metric says *that* something is
slow and a trace says *which*. That is the whole reason tracing is here rather
than more dashboards.

Trace context crosses the boundary because both sides are instrumented: the
httpx instrumentation puts a ``traceparent`` header on the outgoing call, and the
FastAPI instrumentation on the model tier reads it and continues the same trace
instead of starting a new one. Both tiers run the same image, so this is one
configuration, not an integration.

**Every failure here is swallowed.** Tracing that is misconfigured, or an
exporter whose collector is gone, must not stop the service answering. Spans are
handed to a batch processor on a background thread, so an unreachable Jaeger
costs a dropped batch and a log line, never a failed request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

LOGGER = logging.getLogger(__name__)

#: Set by the deployment. Empty means tracing is off, which is the right default
#: for a laptop and for the test suite: no collector, no spans, no noise.
ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"

#: Endpoints the FastAPI instrumentation should not trace. Probes run every few
#: seconds forever and would bury the requests anyone cares about; /metrics is
#: Prometheus scraping, which is not application traffic either.
EXCLUDED_URLS = "health,ready,metrics"

_CONFIGURED = False


def is_configured() -> bool:
    """Whether tracing was successfully set up in this process."""
    return _CONFIGURED


def configure(
    app: FastAPI,
    service_name: str,
    *,
    endpoint: str | None = None,
    service_version: str = "0.1.0",
) -> bool:
    """Instrument ``app`` and export spans over OTLP. Returns whether it worked.

    ``endpoint`` is an OTLP/HTTP collector base URL, for example
    ``http://jaeger:4318``. Jaeger accepts OTLP directly, so there is no separate
    collector in this deployment; see docs/observability.md.

    With no endpoint, this does nothing and says so. That is not a failure: it is
    how the service runs locally and under test.
    """
    global _CONFIGURED

    import os

    target = endpoint or os.environ.get(ENV_OTLP_ENDPOINT) or ""
    if not target:
        LOGGER.debug("no OTLP endpoint configured; tracing is off")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )
        provider = TracerProvider(resource=resource)
        # The batch processor is what keeps an unreachable collector off the
        # request path: spans are queued and flushed on a background thread.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{target.rstrip('/')}/v1/traces"))
        )
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app, excluded_urls=EXCLUDED_URLS)
        # The outgoing half. Without this the trace stops at the application
        # tier and the model tier's work appears as an unexplained gap.
        HTTPXClientInstrumentor().instrument()

        _CONFIGURED = True
        LOGGER.info("tracing to %s as %r", target, service_name)
        return True

    except Exception:
        # Deliberately broad. Anything wrong with telemetry setup -- a missing
        # package, a malformed endpoint, an incompatible version -- must leave a
        # working service behind.
        LOGGER.warning("could not configure tracing; continuing without it", exc_info=True)
        return False


def current_trace_id() -> str | None:
    """The active trace id as hex, for correlating a response with a trace."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return format(context.trace_id, "032x")
    except Exception:  # pragma: no cover
        return None


def add_span_attributes(**attributes: Any) -> None:
    """Annotate the active span, if there is one.

    Only low-cardinality, non-sensitive facts belong here: how many applications
    were in the batch, which model version answered. Never an application's
    field values -- a trace backend is not the place for someone's loan.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None or not span.get_span_context().is_valid:
            return
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:  # pragma: no cover
        LOGGER.debug("could not set span attributes", exc_info=True)
