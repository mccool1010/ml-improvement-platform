"""Tests for metrics and tracing.

Two properties matter more than the rest.

**Cardinality.** A label carrying a request id, a raw path or an application's
field values turns one metric into unbounded many and eventually takes the
metrics server down. The tests here assert that the route label is a template and
that an unmatched path collapses to a constant.

**Telemetry must not be able to break the service.** A platform whose
observability can fail the thing it observes is worse than one with none, so
every failure path is exercised: no collector configured, an unreachable
collector, and instrumentation that raises.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from ml_platform.observability import API_SERVICE, metrics, tracing

SERVICE = "test-service"


def _value(name: str, **labels: str) -> float:
    """One metric sample, or 0.0 when the series does not exist yet."""
    found = REGISTRY.get_sample_value(name, labels)
    return 0.0 if found is None else float(found)


@pytest.fixture
def app() -> FastAPI:
    """A minimal app with the middleware and /metrics installed."""
    application = FastAPI()

    @application.get("/echo/{item}")
    def echo(item: str) -> dict[str, str]:
        return {"item": item}

    @application.get("/boom")
    def boom() -> dict[str, str]:
        raise RuntimeError("deliberate")

    metrics.install(application, SERVICE)
    return application


class TestRequestMetrics:
    def test_a_request_is_counted(self, app: FastAPI) -> None:
        before = _value(
            "ml_platform_http_requests_total",
            service=SERVICE,
            method="GET",
            route="/echo/{item}",
            status="200",
        )
        with TestClient(app) as client:
            assert client.get("/echo/abc").status_code == 200
        after = _value(
            "ml_platform_http_requests_total",
            service=SERVICE,
            method="GET",
            route="/echo/{item}",
            status="200",
        )
        assert after == before + 1

    def test_latency_is_recorded(self, app: FastAPI) -> None:
        before = _value(
            "ml_platform_http_request_duration_seconds_count",
            service=SERVICE,
            method="GET",
            route="/echo/{item}",
        )
        with TestClient(app) as client:
            client.get("/echo/abc")
        after = _value(
            "ml_platform_http_request_duration_seconds_count",
            service=SERVICE,
            method="GET",
            route="/echo/{item}",
        )
        assert after == before + 1

    def test_an_error_is_counted_by_class(self, app: FastAPI) -> None:
        before = _value(
            "ml_platform_http_request_errors_total",
            service=SERVICE,
            method="GET",
            route="unmatched",
            status_class="4xx",
        )
        with TestClient(app) as client:
            assert client.get("/definitely-not-a-route").status_code == 404
        after = _value(
            "ml_platform_http_request_errors_total",
            service=SERVICE,
            method="GET",
            route="unmatched",
            status_class="4xx",
        )
        assert after == before + 1

    def test_an_unhandled_exception_is_counted_as_a_500(self, app: FastAPI) -> None:
        """The most important bar on the error graph must not be the one that
        goes missing because the handler blew up."""
        before = _value(
            "ml_platform_http_request_errors_total",
            service=SERVICE,
            method="GET",
            route="/boom",
            status_class="5xx",
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/boom").status_code == 500
        after = _value(
            "ml_platform_http_request_errors_total",
            service=SERVICE,
            method="GET",
            route="/boom",
            status_class="5xx",
        )
        assert after == before + 1

    def test_the_metrics_endpoint_serves_prometheus_text(self, app: FastAPI) -> None:
        with TestClient(app) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "ml_platform_http_requests_total" in response.text

    def test_the_metrics_endpoint_is_not_in_the_schema(self, app: FastAPI) -> None:
        """It is scrape plumbing, not part of the service's contract."""
        with TestClient(app) as client:
            paths = client.get("/openapi.json").json()["paths"]
        assert "/metrics" not in paths


class TestCardinalityIsBounded:
    """The failure that takes a metrics server down is never a wrong number; it
    is a label somebody let grow without limit."""

    def test_the_route_label_is_a_template_not_a_path(self, app: FastAPI) -> None:
        """A thousand item ids must be one series, not a thousand."""
        with TestClient(app) as client:
            for item in ("a", "b", "c"):
                client.get(f"/echo/{item}")
        series = {
            sample.labels["route"]
            for metric in REGISTRY.collect()
            if metric.name == "ml_platform_http_requests"
            for sample in metric.samples
            if sample.labels.get("service") == SERVICE
        }
        assert "/echo/{item}" in series
        for name in ("/echo/a", "/echo/b", "/echo/c"):
            assert name not in series

    def test_an_unmatched_path_collapses_to_a_constant(self, app: FastAPI) -> None:
        """The path of a 404 is whatever the caller sent, so it is never a label."""
        with TestClient(app) as client:
            client.get("/nope/one")
            client.get("/nope/two")
        series = {
            sample.labels["route"]
            for metric in REGISTRY.collect()
            if metric.name == "ml_platform_http_requests"
            for sample in metric.samples
            if sample.labels.get("service") == SERVICE
        }
        assert metrics.UNMATCHED_ROUTE in series
        assert not any(name.startswith("/nope") for name in series)

    def test_status_classes_are_four_values_not_six_hundred(self) -> None:
        assert metrics._status_class(200) == "2xx"
        assert metrics._status_class(404) == "4xx"
        assert metrics._status_class(503) == "5xx"

    def test_no_metric_declares_an_unbounded_label(self) -> None:
        """A guard against the next person adding one."""
        forbidden = {"request_id", "trace_id", "path", "url", "user", "application", "instance_id"}
        for metric in REGISTRY.collect():
            if not metric.name.startswith("ml_platform_"):
                continue
            for sample in metric.samples:
                assert not (set(sample.labels) & forbidden), (metric.name, sample.labels)


class TestModelTierMetrics:
    def test_outcomes_are_counted_separately(self) -> None:
        """An upstream timeout and an upstream 500 are different operational
        events and must not be summed into one 'error' bar."""
        for outcome in (
            metrics.OUTCOME_SUCCESS,
            metrics.OUTCOME_ERROR,
            metrics.OUTCOME_UNAVAILABLE,
        ):
            before = _value(
                "ml_platform_model_tier_requests_total", service=SERVICE, outcome=outcome
            )
            metrics.observe_model_tier(SERVICE, outcome, 0.01)
            after = _value(
                "ml_platform_model_tier_requests_total", service=SERVICE, outcome=outcome
            )
            assert after == before + 1

    def test_upstream_duration_is_recorded(self) -> None:
        before = _value("ml_platform_model_tier_duration_seconds_count", service=SERVICE)
        metrics.observe_model_tier(SERVICE, metrics.OUTCOME_SUCCESS, 0.02)
        after = _value("ml_platform_model_tier_duration_seconds_count", service=SERVICE)
        assert after == before + 1

    def test_predictions_scored_and_flagged_are_counted(self) -> None:
        scored_before = _value("ml_platform_applications_scored_total", service=SERVICE)
        flagged_before = _value("ml_platform_applications_flagged_total", service=SERVICE)
        metrics.observe_prediction(SERVICE, metrics.OUTCOME_SUCCESS, scored=3, flagged=1)
        assert _value("ml_platform_applications_scored_total", service=SERVICE) == scored_before + 3
        assert (
            _value("ml_platform_applications_flagged_total", service=SERVICE) == flagged_before + 1
        )

    def test_readiness_is_published(self) -> None:
        metrics.set_model_ready(SERVICE, True)
        assert _value("ml_platform_model_ready", service=SERVICE) == 1.0
        metrics.set_model_ready(SERVICE, False)
        assert _value("ml_platform_model_ready", service=SERVICE) == 0.0

    def test_model_info_replaces_rather_than_accumulates(self) -> None:
        """Leaving the old series behind would show two production models at
        once after a promotion."""
        metrics.set_model_info(SERVICE, "m", "1", "production", "kserve")
        metrics.set_model_info(SERVICE, "m", "2", "production", "kserve")
        assert _value(
            "ml_platform_model_info",
            service=SERVICE,
            model_name="m",
            version="1",
            alias="production",
            served_by="kserve",
        ) == pytest.approx(0.0)
        assert _value(
            "ml_platform_model_info",
            service=SERVICE,
            model_name="m",
            version="2",
            alias="production",
            served_by="kserve",
        ) == pytest.approx(1.0)


class TestTelemetryCannotBreakTheService:
    """Every one of these is a way telemetry could take the service down, and
    must not."""

    def test_tracing_is_off_without_an_endpoint(self) -> None:
        """Not an error. It is how the service runs locally and under test."""
        assert tracing.configure(FastAPI(), SERVICE, endpoint="") is False

    def test_a_broken_endpoint_does_not_raise(self) -> None:
        assert tracing.configure(FastAPI(), SERVICE, endpoint="not-a-url::::") in (True, False)

    def test_an_unreachable_collector_still_serves_requests(self) -> None:
        """The batch processor exports on a background thread, so a collector
        that is not there costs a dropped batch, never a failed request."""
        application = FastAPI()

        @application.get("/ping")
        def ping() -> dict[str, str]:
            return {"pong": "yes"}

        metrics.install(application, SERVICE)
        tracing.configure(application, SERVICE, endpoint="http://127.0.0.1:1")

        with TestClient(application) as client:
            assert client.get("/ping").status_code == 200

    def test_a_failing_metric_does_not_reach_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"metrics backend is on fire ({len(args)}, {len(kwargs)})")

        monkeypatch.setattr(metrics.REQUESTS, "labels", _explode)
        metrics.observe_request(SERVICE, "GET", "/x", 200, 0.01)
        metrics.observe_prediction(SERVICE, metrics.OUTCOME_SUCCESS, 1, 0)

    def test_span_attributes_are_safe_without_a_tracer(self) -> None:
        tracing.add_span_attributes(**{"ml.batch.size": 3})

    def test_the_trace_id_is_absent_rather_than_wrong(self) -> None:
        assert tracing.current_trace_id() in (None,) or isinstance(tracing.current_trace_id(), str)


class TestTheApplicationIsInstrumented:
    def test_the_api_exposes_metrics(self) -> None:
        from ml_platform.api.main import create_app

        with TestClient(create_app(load_on_startup=False)) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "ml_platform_http_requests_total" in response.text

    def test_the_predictor_exposes_metrics(self) -> None:
        from ml_platform.serving.predictor import ModelState, create_app

        with TestClient(create_app(ModelState(), load_on_startup=False)) as client:
            response = client.get("/metrics")
        assert response.status_code == 200

    def test_probes_and_scrapes_are_excluded_from_traces(self) -> None:
        """Probes run every few seconds forever and would bury real traffic."""
        for endpoint in ("health", "ready", "metrics"):
            assert endpoint in tracing.EXCLUDED_URLS

    def test_the_api_service_label_is_stable(self) -> None:
        """Dashboards and alerts are written against it."""
        assert API_SERVICE == "inference-api"
