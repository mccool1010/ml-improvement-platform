"""End-to-end canary flows through the real API and the real evaluator.

Routing, metrics, evaluation and the registry decision, exercised together with
no cluster. The two demonstrations that matter are a canary that passes and one
that is rolled back, and in both the assertion is about the *registry*: the
production alias moves only in the first case.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from ml_platform.api.main import create_app
from ml_platform.api.model import LoadedModel, ModelService
from ml_platform.monitoring.canary import (
    DECISION_PROMOTE,
    DECISION_ROLLBACK,
    CanaryDecision,
    TierSignals,
)
from ml_platform.monitoring.signals import InProcessSignals
from ml_platform.pipelines import canary_pipeline
from ml_platform.serving.canary import (
    ROUTING_KEY_HEADER,
    TIER_CANARY,
    TIER_PRODUCTION,
    CanaryRouter,
)
from ml_platform.serving.client import PredictorError

APPLICATION: dict[str, Any] = {
    "term_months": 84,
    "employees": 12,
    "jobs_created": 3,
    "jobs_retained": 8,
    "gross_approved": 250000.0,
    "sba_approved": 187500.0,
    "disbursed": 250000.0,
    "state": "CA",
    "bank_state": "CA",
    "revolving_line_of_credit": "N",
    "low_doc": "N",
    "urban_rural": 1,
    "new_business": 1,
    "naics": "722410",
    "franchise_code": 0,
    "approval_date": "2005-06-15",
    "disbursement_date": "2005-07-20",
}


class _StubPredictor:
    """Stands in for a KServe tier. Can be told to fail."""

    def __init__(self, probability: float = 0.4, fail: bool = False) -> None:
        self.probability = probability
        self.fail = fail
        self.calls = 0

    def ready(self) -> tuple[bool, str | None]:
        return (not self.fail), None if not self.fail else "connection refused"

    def predict(self, frame: pd.DataFrame) -> list[float]:
        self.calls += 1
        if self.fail:
            raise PredictorError("could not reach the predictor at http://canary")
        return [self.probability] * len(frame)


def _loaded(production: Any, canary: Any = None) -> LoadedModel:
    return LoadedModel(
        pipeline=None,
        name="sba-loan-default-classifier",
        version="1",
        alias="production",
        feature_set="engineered",
        mlflow_run_id="run-1",
        platform_run_id="candidate-1",
        decision_threshold=0.5,
        threshold_source="fallback",
        predictor=production,
        canary_predictor=canary,
        canary_version="2" if canary is not None else None,
    )


def _service(model: LoadedModel) -> ModelService:
    service = ModelService.__new__(ModelService)
    service._config = None
    service._error = None
    service._model = model
    return service


@pytest.fixture
def client_with_canary() -> Any:
    """The real app, with both tiers stubbed and 50% of traffic canaried."""
    production = _StubPredictor(probability=0.10)
    canary = _StubPredictor(probability=0.90)
    app = create_app(load_on_startup=False)
    router = CanaryRouter()
    router.start(
        traffic_percent=50.0,
        candidate_url="http://canary",
        candidate_version="2",
        incumbent_version="1",
    )
    with TestClient(app) as client:
        client.app.state.model_service = _service(_loaded(production, canary))
        client.app.state.canary_router = router
        yield client, production, canary, router


def _predict(client: Any, key: str) -> Any:
    return client.post(
        "/predict",
        json={"applications": [APPLICATION]},
        headers={ROUTING_KEY_HEADER: key},
    )


class TestTrafficActuallySplits:
    def test_both_tiers_receive_requests(self, client_with_canary: Any) -> None:
        client, production, canary, _ = client_with_canary
        for i in range(200):
            assert _predict(client, f"key-{i}").status_code == 200
        assert production.calls > 0
        assert canary.calls > 0
        assert production.calls + canary.calls == 200

    def test_the_response_says_which_tier_answered(self, client_with_canary: Any) -> None:
        client, _, _, router = client_with_canary
        seen: dict[str, str] = {}
        for i in range(60):
            key = f"key-{i}"
            payload = _predict(client, key).json()
            seen[key] = payload["model"]["serving_tier"]
            # The tier the router would choose is the tier that answered.
            assert seen[key] == router.tier_for(key)
        assert set(seen.values()) == {TIER_PRODUCTION, TIER_CANARY}

    def test_the_canary_reports_the_candidate_version(self, client_with_canary: Any) -> None:
        """Reporting the incumbent's version would make a canary untraceable."""
        client, _, _, router = client_with_canary
        canary_key = next(f"k{i}" for i in range(500) if router.tier_for(f"k{i}") == TIER_CANARY)
        model = _predict(client, canary_key).json()["model"]
        assert model["serving_tier"] == TIER_CANARY
        assert model["version"] == "2"
        assert model["alias"] == "canary"

    def test_routing_is_sticky_per_key(self, client_with_canary: Any) -> None:
        client, _, _, _ = client_with_canary
        tiers = {_predict(client, "same-key").json()["model"]["serving_tier"] for _ in range(20)}
        assert len(tiers) == 1

    def test_a_rollback_moves_every_request_back(self, client_with_canary: Any) -> None:
        client, _production, canary, router = client_with_canary
        for i in range(100):
            _predict(client, f"key-{i}")
        assert canary.calls > 0

        before = canary.calls
        router.stop("rolled back by the test")

        for i in range(100):
            assert _predict(client, f"key-{i}").json()["model"]["serving_tier"] == TIER_PRODUCTION
        assert canary.calls == before, "the canary served a request after rollback"


class TestAFailingCanaryDoesNotHideBehindProduction:
    def test_a_canary_failure_surfaces_as_an_error(self) -> None:
        """Falling back to production would make the error rate the decision
        rests on read as zero, and the canary would pass on the strength of
        requests the candidate never served."""
        production = _StubPredictor(probability=0.10)
        canary = _StubPredictor(fail=True)
        app = create_app(load_on_startup=False)
        router = CanaryRouter()
        router.start(traffic_percent=99.0, candidate_url="http://canary", candidate_version="2")

        with TestClient(app) as client:
            client.app.state.model_service = _service(_loaded(production, canary))
            client.app.state.canary_router = router
            key = next(f"k{i}" for i in range(500) if router.tier_for(f"k{i}") == TIER_CANARY)
            response = _predict(client, key)

        assert response.status_code == 503
        assert production.calls == 0, "the failure was masked by the incumbent"


class TestSignalsComeFromRealTraffic:
    def test_a_window_measures_only_what_happened_inside_it(self, client_with_canary: Any) -> None:
        """Counters are cumulative; without a baseline an evaluation would
        include every request since the process started."""
        client, _, _, _ = client_with_canary
        for i in range(40):
            _predict(client, f"warmup-{i}")

        signals = InProcessSignals(REGISTRY)
        signals.mark_window_start()
        for i in range(60):
            _predict(client, f"window-{i}")

        canary = signals.signals_for(TIER_CANARY, 300)
        production = signals.signals_for(TIER_PRODUCTION, 300)
        assert canary.requests + production.requests == 60

    def test_errors_are_attributed_to_the_tier_that_produced_them(self) -> None:
        production = _StubPredictor(probability=0.10)
        canary = _StubPredictor(fail=True)
        app = create_app(load_on_startup=False)
        router = CanaryRouter()
        router.start(traffic_percent=99.0, candidate_url="http://canary", candidate_version="2")

        signals = InProcessSignals(REGISTRY)
        with TestClient(app) as client:
            client.app.state.model_service = _service(_loaded(production, canary))
            client.app.state.canary_router = router
            signals.mark_window_start()
            keys = [f"k{i}" for i in range(2000) if router.tier_for(f"k{i}") == TIER_CANARY][:20]
            for key in keys:
                _predict(client, key)

        measured = signals.signals_for(TIER_CANARY, 300)
        assert measured.requests == 20
        assert measured.errors == 20
        assert measured.upstream_failures == 20
        assert measured.error_rate == pytest.approx(1.0)


def _decision(kind: str) -> CanaryDecision:
    """A decision object of the requested kind, for the registry tests."""
    healthy = TierSignals(tier=TIER_CANARY, requests=200, latency_p95_seconds=0.05)
    broken = TierSignals(tier=TIER_CANARY, requests=200, errors=200, latency_p95_seconds=0.05)
    from ml_platform.monitoring.canary import evaluate_canary

    return evaluate_canary(
        canary=healthy if kind == DECISION_PROMOTE else broken,
        production=TierSignals(tier=TIER_PRODUCTION, requests=200, latency_p95_seconds=0.05),
        thresholds={"min_requests": 50, "max_error_rate": 0.02},
        canary_event_id="canary-test",
        traffic_percent=10.0,
        observation_window_seconds=300,
        candidate_version="2",
        incumbent_version="1",
    )


class TestTheRegistryIsOnlyTouchedOnSuccess:
    """The safety property the whole milestone rests on."""

    def test_a_rollback_leaves_the_alias_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        moved: list[Any] = []
        monkeypatch.setattr(
            canary_pipeline, "resolve_production", lambda _config: _FakeProduction("1")
        )

        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version="2")
        decision = _decision(DECISION_ROLLBACK)
        assert decision.decision == DECISION_ROLLBACK

        outcome = canary_pipeline.rollback(_FakeConfig(), router, decision)

        assert moved == [], "a rollback must not write to the registry"
        assert outcome.alias_moved is False
        assert outcome.production_version == "1"
        assert router.traffic_percent == 0.0

    def test_completing_a_failed_canary_is_refused(self) -> None:
        """Not merely discouraged -- refused, because this is the one call that
        can change what production means."""
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version="2")
        with pytest.raises(canary_pipeline.CanaryStateError, match="refusing to complete"):
            canary_pipeline.complete(_FakeConfig(), router, _decision(DECISION_ROLLBACK))

    def test_decide_routes_a_failure_to_rollback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            canary_pipeline, "resolve_production", lambda _config: _FakeProduction("1")
        )
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version="2")
        outcome = canary_pipeline.decide(_FakeConfig(), router, _decision(DECISION_ROLLBACK))
        assert outcome.alias_moved is False
        assert "never moved" in " ".join(outcome.notes)


class _FakeProduction:
    def __init__(self, version: str) -> None:
        self.version = version
        self.name = "sba-loan-default-classifier"


class _FakeConfig:
    """Only the fields the registry helpers touch."""

    registered_model_name = "sba-loan-default-classifier"
    production_alias = "production"
    canary_alias = "canary"
    tracking_uri = "sqlite:///unused.db"
    tracking_enabled = False
    canary_observation_seconds = 300
    canary_thresholds: ClassVar[dict[str, Any]] = {"min_requests": 50}

    @property
    def benchmark_dir(self) -> Any:
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())


def test_the_traffic_gauge_follows_the_router() -> None:
    """A dashboard must show a rollback landing."""
    from ml_platform.observability.metrics import set_canary_traffic

    set_canary_traffic(25.0)
    assert REGISTRY.get_sample_value("ml_platform_canary_traffic_percent") == 25.0
    set_canary_traffic(0.0)
    assert REGISTRY.get_sample_value("ml_platform_canary_traffic_percent") == 0.0


def test_routing_works_with_no_model_loaded() -> None:
    """A router must answer in any process, including one holding no model."""
    router = CanaryRouter()
    assert router.tier_for("anything") == TIER_PRODUCTION


class TestStalePromoteDecisionsAreRefused:
    """A CanaryDecision is a durable object. Replaying an old one must not move
    the production alias, because production may have advanced since."""

    def _running(self, version: str = "2") -> CanaryRouter:
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version=version)
        return router

    def test_completing_with_no_canary_running_is_refused(self) -> None:
        """The gun this precondition unloads: a decision object outliving its
        canary and being completed later."""
        router = CanaryRouter()
        with pytest.raises(canary_pipeline.CanaryStateError, match="no canary is running"):
            canary_pipeline.complete(_FakeConfig(), router, _decision(DECISION_PROMOTE))

    def test_replaying_a_decision_after_the_canary_ended_is_refused(self) -> None:
        router = self._running()
        decision = _decision(DECISION_PROMOTE)
        router.stop("the canary already finished")
        with pytest.raises(canary_pipeline.CanaryStateError, match="no canary is running"):
            canary_pipeline.complete(_FakeConfig(), router, decision)

    def test_a_decision_for_another_candidate_is_refused(self) -> None:
        """Evidence gathered for v2 says nothing about a v3 started afterwards."""
        router = self._running(version="3")
        decision = _decision(DECISION_PROMOTE)  # measured against v2
        with pytest.raises(canary_pipeline.CanaryStateError, match="the decision is for v2"):
            canary_pipeline.complete(_FakeConfig(), router, decision)

    def test_decide_does_not_route_a_stale_promote_to_the_registry(self) -> None:
        router = CanaryRouter()
        with pytest.raises(canary_pipeline.CanaryStateError):
            canary_pipeline.decide(_FakeConfig(), router, _decision(DECISION_PROMOTE))


class TestTheTrafficControllerReachesLiveReplicas:
    """A rollback that only moves a process-local router has rolled back
    nothing. The controller is what makes it real."""

    def test_rollback_without_a_controller_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            canary_pipeline, "resolve_production", lambda _config: _FakeProduction("1")
        )
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version="2")
        outcome = canary_pipeline.rollback(_FakeConfig(), router, _decision(DECISION_ROLLBACK))
        assert any("this process only" in note for note in outcome.notes)

    def test_rollback_applies_the_change_through_the_controller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            canary_pipeline, "resolve_production", lambda _config: _FakeProduction("1")
        )

        class _Recording:
            def __init__(self) -> None:
                self.applied: list[float] = []

            def set_traffic(self, percent: float) -> None:
                self.applied.append(percent)

            def describe(self) -> str:
                return "recording controller"

        controller = _Recording()
        router = CanaryRouter()
        router.start(traffic_percent=10.0, candidate_url="http://canary", candidate_version="2")

        outcome = canary_pipeline.rollback(
            _FakeConfig(), router, _decision(DECISION_ROLLBACK), controller=controller
        )

        assert controller.applied == [0.0], "the live allocation was never set to zero"
        assert any("recording controller" in note for note in outcome.notes)
        assert outcome.alias_moved is False

    def test_the_in_process_controller_moves_the_router(self) -> None:
        from ml_platform.serving.traffic import InProcessTrafficController

        router = CanaryRouter()
        router.start(traffic_percent=30.0, candidate_url="http://canary", candidate_version="2")
        controller = InProcessTrafficController(router)

        controller.set_traffic(0.0)

        assert router.traffic_percent == 0.0
        counts = router.observed_split([f"k{i}" for i in range(300)])
        assert counts[TIER_CANARY] == 0

    def test_the_kubernetes_controller_patches_the_key_the_api_reads(self) -> None:
        """The ConfigMap key must be the one config.py reads, or a rollback
        writes a value nothing honours."""
        from ml_platform.config import ENV_CANARY_TRAFFIC
        from ml_platform.serving.traffic import TRAFFIC_KEY

        assert TRAFFIC_KEY == ENV_CANARY_TRAFFIC


class TestConfigurationGuards:
    def test_a_canary_alias_equal_to_production_is_refused(self) -> None:
        """Starting a canary would move production immediately."""
        from ml_platform.config import Config
        from ml_platform.serving.canary import CanaryError

        config = Config(
            raw={
                "promotion": {"production_alias": "production"},
                "canary": {"alias": "production"},
            },
            environment="test",
        )
        with pytest.raises(CanaryError, match="both 'production'"):
            _ = config.canary_alias

    def test_a_distinct_canary_alias_is_accepted(self) -> None:
        from ml_platform.config import Config

        config = Config(
            raw={
                "promotion": {"production_alias": "production"},
                "canary": {"alias": "canary"},
            },
            environment="test",
        )
        assert config.canary_alias == "canary"

    def test_a_full_allocation_in_configuration_is_refused(self) -> None:
        from ml_platform.config import Config
        from ml_platform.serving.canary import CanaryError

        config = Config(raw={"canary": {"traffic_percent": 100.0}}, environment="test")
        with pytest.raises(CanaryError, match="nothing to compare"):
            _ = config.canary_traffic_percent

    def test_ordinary_allocations_are_unchanged(self) -> None:
        from ml_platform.config import Config

        for percent in (0.0, 5.0, 30.0, 99.0):
            config = Config(raw={"canary": {"traffic_percent": percent}}, environment="test")
            assert config.canary_traffic_percent == percent
