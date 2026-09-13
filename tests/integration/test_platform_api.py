"""Tests for the read-only platform API the dashboard consumes.

The property that matters most here is negative: **the API must never invent
platform state.** A dashboard showing a plausible model version, drift result or
latency figure that came from a default rather than a system is worse than one
showing nothing, because a reader cannot tell which they are looking at. So most
of these tests run with the dependency deliberately unreachable and assert that
the response says so.

The second property is that the aggregation stays an aggregation. It must not
serve a model, hold state, decide anything, or become a second source of truth
about what production is.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml_platform.api.main import create_app
from ml_platform.api.platform import (
    STATUS_DEGRADED,
    STATUS_HEALTHY,
    STATUS_UNAVAILABLE,
)
from ml_platform.config import Config

#: A tracking URI that cannot resolve, so MLflow is genuinely unreachable
#: rather than mocked away.
DEAD_TRACKING_URI = "http://127.0.0.1:1"


def _config(**overrides: Any) -> Config:
    from ml_platform.config import load_config

    base = load_config("production")
    return Config(raw={**base.raw, **overrides}, environment="test")


@pytest.fixture(autouse=True)
def _fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the MLflow client's retry budget for the whole module.

    The aggregation layer already bounds it so a browser is not left on a
    spinner, but each unreachable call still costs one socket timeout, and
    these tests make many. Tightening it here keeps the suite quick without
    changing what is being tested: the dependency is still genuinely dead.
    """
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")


@pytest.fixture
def offline_client() -> Any:
    """The app with every external dependency pointed at nothing.

    No mocks: MLflow and Prometheus are given addresses that refuse
    connections, which is the same thing the pod experiences when they are down.
    """
    config = _config(
        tracking={"enabled": True, "backend_uri": DEAD_TRACKING_URI, "experiment_name": "x"},
        dashboard={"prometheus_url": DEAD_TRACKING_URI},
    )
    with TestClient(create_app(config, load_on_startup=False)) as client:
        yield client


@pytest.fixture
def local_client() -> Any:
    """The app against the project's real local store, whatever it holds."""
    with TestClient(create_app(load_on_startup=False)) as client:
        yield client


ENDPOINTS = [
    "/platform/health",
    "/platform/model",
    "/platform/lifecycle",
    "/platform/promotions",
    "/platform/drift",
    "/platform/canary",
    "/platform/failure",
    "/platform/observability",
    "/platform/gates",
    "/platform/events",
]


class TestEveryEndpointAnswers:
    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_it_returns_200_even_with_nothing_behind_it(
        self, offline_client: Any, path: str
    ) -> None:
        """A dead dependency is a reportable state, not a server error. A 500
        here would make the dashboard unable to distinguish "MLflow is down"
        from "the platform API is broken"."""
        response = offline_client.get(path)
        assert response.status_code == 200, response.text

    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_the_response_matches_its_declared_schema(self, offline_client: Any, path: str) -> None:
        """FastAPI validates against the response_model on the way out, so a
        200 here is itself the schema assertion; this checks the shape is what
        the dashboard is written against."""
        payload = offline_client.get(path).json()
        assert isinstance(payload, list | dict)

    def test_the_endpoints_are_documented(self, local_client: Any) -> None:
        paths = local_client.get("/openapi.json").json()["paths"]
        for path in ENDPOINTS:
            assert path in paths, path

    @pytest.mark.parametrize("path", ENDPOINTS)
    def test_they_are_read_only(self, offline_client: Any, path: str) -> None:
        """No control operation was needed, so none is exposed. A POST here
        would be a way to bypass the CLI safety checks."""
        assert offline_client.post(path).status_code == 405


class TestNothingIsFabricated:
    """The central requirement."""

    def test_the_model_reports_unavailable_rather_than_a_plausible_version(
        self, offline_client: Any
    ) -> None:
        payload = offline_client.get("/platform/model").json()
        assert payload["available"] is False
        assert payload["detail"], "an unavailable section must say why"
        assert payload["version"] is None
        assert payload["name"] is None
        assert payload["metrics"] == {}

    def test_promotions_reports_unavailable_rather_than_an_empty_success(
        self, offline_client: Any
    ) -> None:
        payload = offline_client.get("/platform/promotions").json()
        assert payload["available"] is False
        assert payload["versions"] == []
        assert payload["production_version"] is None

    def test_drift_reports_unavailable_rather_than_normal(self, offline_client: Any) -> None:
        """The dangerous default: reporting "no drift" when the truth is "we
        cannot tell". One reads as a healthy system."""
        payload = offline_client.get("/platform/drift").json()
        assert payload["available"] is False
        assert payload["decision"] is None
        assert payload["drift_detected"] is None
        assert payload["n_drifted"] is None

    def test_observability_reports_unavailable_rather_than_zero_latency(
        self, offline_client: Any
    ) -> None:
        """Zero latency and no traffic look like a fast, idle service. They are
        not the same as an unreachable Prometheus."""
        payload = offline_client.get("/platform/observability").json()
        assert payload["available"] is False
        assert payload["detail"]
        for field in (
            "request_rate",
            "error_ratio",
            "latency_p50_seconds",
            "latency_p95_seconds",
        ):
            assert payload[field] is None, field

    def test_events_are_empty_rather_than_invented(self, offline_client: Any) -> None:
        assert offline_client.get("/platform/events").json() == []

    def test_canary_reports_no_traffic_rather_than_a_guess(self, offline_client: Any) -> None:
        """Zero is the true allocation here, read from configuration, not a
        placeholder: the router is built from config at startup."""
        payload = offline_client.get("/platform/canary").json()
        assert payload["traffic_percent"] == 0.0
        assert payload["active"] is False
        assert payload["last_decision"] is None


class TestHealthIsHonestAboutDegradation:
    def test_a_dead_dependency_is_named(self, offline_client: Any) -> None:
        payload = offline_client.get("/platform/health").json()
        by_name = {c["name"]: c for c in payload["components"]}
        assert by_name["MLflow"]["status"] == STATUS_UNAVAILABLE
        assert by_name["Prometheus"]["status"] == STATUS_UNAVAILABLE
        assert by_name["API"]["status"] == STATUS_HEALTHY

    def test_every_component_says_what_it_owns(self, offline_client: Any) -> None:
        """So a reader knows what a component being down actually costs."""
        for component in offline_client.get("/platform/health").json()["components"]:
            assert component["owns"], component["name"]

    def test_no_model_means_the_platform_cannot_serve(self, offline_client: Any) -> None:
        payload = offline_client.get("/platform/health").json()
        assert payload["can_serve"] is False
        assert payload["status"] == STATUS_UNAVAILABLE

    def test_serving_with_a_dead_registry_is_degraded_not_unavailable(self) -> None:
        """The M13/M15 finding, surfaced: MLflow is not on the request path, so
        losing it degrades the platform without stopping predictions."""
        from ml_platform.api.model import LoadedModel, ModelService

        config = _config(
            tracking={"enabled": True, "backend_uri": DEAD_TRACKING_URI, "experiment_name": "x"},
            dashboard={"prometheus_url": DEAD_TRACKING_URI},
        )
        service = ModelService.__new__(ModelService)
        service._config = config
        service._error = None
        service._model = LoadedModel(
            pipeline=object(),
            name="sba-loan-default-classifier",
            version="1",
            alias="production",
            feature_set="engineered",
            mlflow_run_id="run",
            platform_run_id="candidate",
            decision_threshold=0.5,
            threshold_source="fallback",
        )

        with TestClient(create_app(config, load_on_startup=False)) as client:
            client.app.state.model_service = service
            payload = client.get("/platform/health").json()

        assert payload["can_serve"] is True
        assert payload["status"] == STATUS_DEGRADED


class TestTheLifecycleMatchesTheCode:
    def test_every_stage_names_a_real_component_and_milestone(self, offline_client: Any) -> None:
        stages = offline_client.get("/platform/lifecycle").json()
        assert len(stages) >= 13
        for stage in stages:
            assert stage["stage"] and stage["component"] and stage["milestone"]
            assert stage["state"] in {"observed", "implemented", "untracked"}

    def test_the_order_is_the_lifecycle_order(self, offline_client: Any) -> None:
        names = [s["stage"] for s in offline_client.get("/platform/lifecycle").json()]
        for earlier, later in (
            ("train", "experiment"),
            ("quality gates", "promote / reject"),
            ("deploy", "monitor"),
            ("drift", "retrain"),
            ("canary", "rollback"),
        ):
            assert names.index(earlier) < names.index(later), f"{earlier} before {later}"

    def test_stages_with_no_evidence_are_implemented_not_observed(
        self, offline_client: Any
    ) -> None:
        """With MLflow down nothing can be observed, and the dashboard must not
        claim otherwise."""
        stages = {s["stage"]: s["state"] for s in offline_client.get("/platform/lifecycle").json()}
        assert stages["drift"] == "implemented"
        assert stages["canary"] == "implemented"
        assert stages["optimize"] == "implemented"

    def test_failure_recovery_is_untracked_rather_than_unexercised(self, local_client: Any) -> None:
        """Failure reports are not written to MLflow, so this view cannot know
        whether the scenarios ran. "implemented" would claim they never did."""
        stages = {s["stage"]: s["state"] for s in local_client.get("/platform/lifecycle").json()}
        assert stages["failure / recovery"] == "untracked"

    def test_optimize_is_observed_when_its_experiment_has_runs(
        self, monkeypatch: pytest.MonkeyPatch, local_client: Any
    ) -> None:
        """Optimisation runs live in their own experiment; the stage must look
        there rather than stay permanently hollow."""
        from ml_platform.api import platform

        monkeypatch.setattr(platform._Mlflow, "has_experiment_runs", lambda _self, _name: True)
        stages = {s["stage"]: s["state"] for s in local_client.get("/platform/lifecycle").json()}
        assert stages["optimize"] == "observed"


class TestTheProductionAliasIsShown:
    def test_the_production_row_lists_its_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Some backends return searched versions with empty `aliases`. The row
        marked production must still show the alias, not a dash."""
        from types import SimpleNamespace

        from ml_platform.api import platform

        version = SimpleNamespace(
            version="1", run_id="r1", tags={}, aliases=[], creation_timestamp=0
        )

        class FakeClient:
            def get_model_version_by_alias(self, _name: str, _alias: str) -> Any:
                return version

        monkeypatch.setattr(platform._Mlflow, "versions", lambda _self: [version])
        monkeypatch.setattr(platform._Mlflow, "client", lambda _self: FakeClient())
        monkeypatch.setattr(platform._Mlflow, "training_runs", lambda _self, _limit=200: [])

        with TestClient(create_app(load_on_startup=False)) as client:
            payload = client.get("/platform/promotions").json()
        row = payload["versions"][0]
        assert row["is_production"] is True
        assert "production" in row["aliases"]


class TestTheClaimsTheProjectMustNotBlur:
    def test_drift_states_that_no_performance_signal_exists(self, offline_client: Any) -> None:
        """On the payload, not only in the docs: a dashboard that implied a live
        accuracy signal would undo the three-clocks rule."""
        payload = offline_client.get("/platform/drift").json()
        assert payload["performance_signal_available"] is False
        assert "matured labels" in payload["performance_note"]

    def test_canary_states_that_accuracy_is_not_a_signal(self, offline_client: Any) -> None:
        payload = offline_client.get("/platform/canary").json()
        assert payload["accuracy_used_as_signal"] is False
        assert "Accuracy is excluded" in payload["signal_note"]
        assert payload["rollback_signals"], "the real signals must be listed"
        joined = " ".join(payload["rollback_signals"]).lower()
        for word in ("accuracy", "precision", "roc"):
            assert word not in joined


class TestTheFailureCatalogueComesFromTheHarness:
    def test_it_lists_exactly_the_scenarios_that_exist(self, offline_client: Any) -> None:
        """Read from the runner's own registry, so the dashboard cannot show a
        scenario nobody implemented."""
        from ml_platform.failure.runner import ALL_SCENARIOS

        payload = offline_client.get("/platform/failure").json()
        assert {s["scenario"] for s in payload["scenarios"]} == set(ALL_SCENARIOS)
        assert len(payload["scenarios"]) == 6

    def test_the_double_is_labelled_as_one(self, offline_client: Any) -> None:
        payload = offline_client.get("/platform/failure").json()
        modes = {s["scenario"]: s["mode"] for s in payload["scenarios"]}
        assert modes["bad_candidate_promotion"] == "controlled-double"
        assert modes["model_serving_failure"] == "live-kubernetes"

    def test_it_does_not_claim_live_execution(self, offline_client: Any) -> None:
        """The harness writes evidence where it runs, which is not in a pod."""
        payload = offline_client.get("/platform/failure").json()
        assert payload["evidence_available"] is False
        assert payload["detail"]
        for scenario in payload["scenarios"]:
            assert scenario["passed"] is None


class TestNothingSensitiveIsExposed:
    def test_no_response_leaks_a_filesystem_path_or_credential(self, local_client: Any) -> None:
        """The tracking URI contains an absolute local path; it must not travel
        to a browser, and neither must anything credential-shaped."""
        forbidden = ("sqlite:///", "C:/", "C:\\", "password", "secret", "token=", "Bearer ")
        for path in ENDPOINTS:
            body = local_client.get(path).text
            for needle in forbidden:
                assert needle not in body, f"{path} leaked {needle!r}"

    def test_the_model_section_exposes_identity_not_location(self, local_client: Any) -> None:
        payload = local_client.get("/platform/model").json()
        assert "artifact_uri" not in payload
        assert "source" not in payload
        assert "tracking_uri" not in payload


class TestTheDashboardMountIsOptional:
    def test_the_api_works_without_a_built_dashboard(self, local_client: Any) -> None:
        """No node toolchain is needed to run or test the platform. Absent a
        build, nothing is mounted and the API is what M7-M15 shipped."""
        assert local_client.get("/health").status_code == 200
        assert local_client.get("/platform/health").status_code == 200

    def test_inference_endpoints_are_unchanged(self, local_client: Any) -> None:
        """M16 adds a router; it does not touch the M7 contract."""
        paths = local_client.get("/openapi.json").json()["paths"]
        for path in ("/health", "/ready", "/model", "/predict"):
            assert path in paths, path


class TestTheCataloguesCannotGoStale:
    """The gate and invariant catalogues are derived, not transcribed.

    A dashboard describing a gate the promotion code no longer runs, or missing
    one it does, is worse than no dashboard: a reader would take it as evidence
    about the system. These tests are what makes "derived from the source
    module" true rather than merely claimed.
    """

    def test_every_configured_gate_is_reported(self, local_client: Any) -> None:
        from ml_platform.promotion.gates import GATE_BUILDERS

        reported = [gate["name"] for gate in local_client.get("/platform/gates").json()]
        assert reported == list(GATE_BUILDERS)

    def test_every_gate_has_a_rationale(self, local_client: Any) -> None:
        for gate in local_client.get("/platform/gates").json():
            assert gate["rationale"].strip(), gate["name"]

    def test_no_rationale_describes_a_gate_that_does_not_exist(self) -> None:
        from ml_platform.api.platform import GATE_RATIONALE
        from ml_platform.promotion.gates import GATE_BUILDERS

        assert set(GATE_RATIONALE) == set(GATE_BUILDERS)

    def test_the_configured_flag_matches_this_installation(self, local_client: Any) -> None:
        from ml_platform.config import load_config

        config = load_config("production")
        configured = set(config.gate_config)
        for gate in local_client.get("/platform/gates").json():
            assert gate["configured"] == (gate["name"] in configured), gate["name"]

    def test_every_named_invariant_is_reported(self, local_client: Any) -> None:
        from ml_platform.failure import invariants as inv

        defined = {
            value
            for key, value in vars(inv).items()
            if key.isupper() and isinstance(value, str) and not key.startswith("_")
        }
        reported = {
            item["name"] for item in local_client.get("/platform/failure").json()["invariants"]
        }
        assert reported == defined

    def test_no_falsification_describes_an_invariant_that_does_not_exist(self) -> None:
        from ml_platform.api.platform import INVARIANT_FALSIFIED_BY
        from ml_platform.failure import invariants as inv

        defined = {
            value
            for key, value in vars(inv).items()
            if key.isupper() and isinstance(value, str) and not key.startswith("_")
        }
        assert set(INVARIANT_FALSIFIED_BY) == defined

    def test_every_invariant_has_a_falsifying_observation(self, local_client: Any) -> None:
        for item in local_client.get("/platform/failure").json()["invariants"]:
            assert item["falsified_by"].strip(), item["name"]

    def test_the_scenario_catalogue_matches_the_runner(self, local_client: Any) -> None:
        from ml_platform.failure.runner import LIVE_SCENARIOS, OFFLINE_SCENARIOS

        payload = local_client.get("/platform/failure").json()["scenarios"]
        assert {s["scenario"] for s in payload} == set(LIVE_SCENARIOS) | set(OFFLINE_SCENARIOS)
        # A controlled double must never be presented as a live demonstration.
        for scenario in payload:
            expected = (
                "live-kubernetes" if scenario["scenario"] in LIVE_SCENARIOS else "controlled-double"
            )
            assert scenario["mode"] == expected, scenario["scenario"]


class TestTheLatestRetrainingDecisionIsNotHidden:
    def test_a_later_check_without_a_retrain_still_shows_the_last_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newer drift check with no retrain of its own must not blank the most
        recent retraining decision -- and must say which check that decision
        belonged to, rather than implying the latest one caused it."""
        from types import SimpleNamespace

        from ml_platform.api import platform

        def run(run_id: str, tags: dict[str, str], metrics: dict[str, float]) -> Any:
            return SimpleNamespace(
                info=SimpleNamespace(run_id=run_id, start_time=0),
                data=SimpleNamespace(tags=tags, metrics=metrics, params={"threshold_psi": "0.1"}),
            )

        latest_check = run("d2", {"drift_decision": "retrain", "drift_event_id": "drift-new"}, {})
        candidate = run(
            "c1", {"drift_event_id": "drift-old"}, {"validation_average_precision": 0.7344}
        )

        def runs(_self: Any, *, run_type: str | None = None, limit: int = 25) -> list[Any]:
            del limit
            return {"drift_check": [latest_check], "retraining_candidate": [candidate]}.get(
                run_type or "", []
            )

        monkeypatch.setattr(platform._Mlflow, "runs", runs)
        monkeypatch.setattr(platform._Mlflow, "versions", lambda _self: [])
        monkeypatch.setattr(platform._Mlflow, "client", lambda _self: None)

        with TestClient(create_app(load_on_startup=False)) as client:
            retraining = client.get("/platform/drift").json()["retraining"]
        assert retraining["ran"] is True
        assert retraining["linked_to_latest_check"] is False
        assert retraining["triggered_by"] == "drift-old"
        assert retraining["promoted"] is False
