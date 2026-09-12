"""Tests for the failure-engineering harness.

Two things are under test and they are different.

The **invariants** must be right, because every live demonstration is only worth
what its invariant is worth. An invariant that cannot fail proves nothing, so
each one here is exercised in both directions: a system behaving correctly, and
a system doing the exact thing the invariant exists to forbid.

The **harness** must restore what it breaks. A failure exercise that can leave a
cluster broken is a worse liability than the failures it tests for, so the
restore path is tested including the case where the observation itself raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform.failure import invariants as inv
from ml_platform.failure.report import (
    MODE_DOUBLE,
    MODE_LIVE,
    FailureReport,
    ScenarioEvidence,
    new_run_id,
)


def _served(n: int = 5) -> list[dict[str, Any]]:
    return [{"status": 200, "scored": True, "probability": 0.004} for _ in range(n)]


def _refused(n: int = 5, status: int = 503) -> list[dict[str, Any]]:
    return [{"status": status, "scored": False} for _ in range(n)]


class TestNoFabricatedPredictions:
    """The invariant the whole milestone turns on."""

    def test_explicit_failures_satisfy_it(self) -> None:
        """Failing closed is success: 503s are the correct answer here."""
        result = inv.no_fabricated_predictions(_refused())
        assert result.held is True
        assert result.evidence["scored_responses"] == 0

    def test_a_single_fabricated_score_violates_it(self) -> None:
        """The failure mode being excluded: one plausible number from nowhere."""
        responses = [*_refused(4), {"status": 200, "scored": True, "probability": 0.5}]
        result = inv.no_fabricated_predictions(responses)
        assert result.held is False
        assert result.evidence["scored_responses"] == 1

    def test_a_silent_fallback_to_the_incumbent_would_be_caught(self) -> None:
        """If the API answered canary failures from production, every response
        would be a 200 with a score and this would fail."""
        assert inv.no_fabricated_predictions(_served()).held is False


class TestProductionAliasUnchanged:
    def test_an_unchanged_alias_holds(self) -> None:
        assert inv.production_alias_unchanged("1", "1").held is True

    def test_a_moved_alias_violates_it(self) -> None:
        result = inv.production_alias_unchanged("1", "2")
        assert result.held is False
        assert result.evidence == {"before": "1", "after": "2"}

    def test_a_registry_that_went_away_is_not_silently_equal(self) -> None:
        """None before and None after is equal and would hold; None only after
        is a change and must not."""
        assert inv.production_alias_unchanged("1", None).held is False
        assert inv.production_alias_unchanged(None, None).held is True


class TestFailedCandidateNotProduction:
    def test_a_rejected_unregistered_candidate_holds(self) -> None:
        assert inv.failed_candidate_not_production(
            promoted=False, registered=False, alias_before="1", alias_after="1"
        ).held

    @pytest.mark.parametrize(
        ("promoted", "registered", "before", "after"),
        [
            (True, False, "1", "1"),
            (False, True, "1", "1"),
            (False, False, "1", "2"),
        ],
    )
    def test_any_one_of_the_three_failing_violates_it(
        self, promoted: bool, registered: bool, before: str, after: str
    ) -> None:
        assert not inv.failed_candidate_not_production(
            promoted=promoted, registered=registered, alias_before=before, alias_after=after
        ).held


class TestCanaryRollbackRestoresIncumbent:
    def test_zero_canary_requests_after_rollback_holds(self) -> None:
        assert inv.canary_rollback_restores_incumbent(
            canary_before=12, canary_after=0, total_after=40
        ).held

    def test_any_remaining_canary_traffic_violates_it(self) -> None:
        assert not inv.canary_rollback_restores_incumbent(
            canary_before=12, canary_after=1, total_after=40
        ).held

    def test_observing_nothing_is_not_proof(self) -> None:
        """Zero canary requests out of zero requests proves nothing at all."""
        assert not inv.canary_rollback_restores_incumbent(
            canary_before=12, canary_after=0, total_after=0
        ).held


class TestTelemetryFailureIsolated:
    def test_every_request_served_holds(self) -> None:
        assert inv.telemetry_failure_isolated(_served(10)).held

    def test_one_failed_request_violates_it(self) -> None:
        assert not inv.telemetry_failure_isolated([*_served(9), *_refused(1)]).held

    def test_no_requests_is_not_proof(self) -> None:
        assert not inv.telemetry_failure_isolated([]).held


class TestInferenceUnaffected:
    """The invariant the live MLflow scenario reports on.

    Same shape as the telemetry one but its own name, so a report never claims
    "telemetry_failure_isolated" about an exercise that switched off the
    registry. Both directions, because an invariant that cannot fail proves
    nothing.
    """

    def test_inference_still_served_while_the_registry_is_down_holds(self) -> None:
        """The MLflow scenario's actual finding: the model was resolved at
        startup and scoring goes to KServe, so the registry is not on the
        request path and every request is answered normally."""
        result = inv.inference_unaffected(_served(8))
        assert result.held is True
        assert result.name == inv.INFERENCE_UNAFFECTED
        assert result.evidence["served"] == 8
        assert result.evidence["status_codes"] == [200]

    def test_inference_becoming_unavailable_violates_it(self) -> None:
        """If a registry outage did reach the request path, one refused request
        is enough to falsify the claim that inference was unaffected."""
        result = inv.inference_unaffected([*_served(7), *_refused(1)])
        assert result.held is False
        assert result.evidence["served"] == 7
        assert result.evidence["requests"] == 8


class TestRecoveryReturnsKnownState:
    def test_healthy_and_identical_holds(self) -> None:
        identity = {"name": "m", "version": "1", "alias": "production"}
        assert inv.recovery_returns_known_state(
            healthy=True, model_identity_before=identity, model_identity_after=dict(identity)
        ).held

    def test_coming_back_unhealthy_violates_it(self) -> None:
        identity = {"version": "1"}
        assert not inv.recovery_returns_known_state(
            healthy=False, model_identity_before=identity, model_identity_after=identity
        ).held

    def test_coming_back_as_a_different_model_violates_it(self) -> None:
        """The worse of the two failures: healthy, and quietly serving something
        else."""
        assert not inv.recovery_returns_known_state(
            healthy=True,
            model_identity_before={"version": "1"},
            model_identity_after={"version": "2"},
        ).held


class TestDependencyFailureIsExplicit:
    def test_a_clear_failure_holds(self) -> None:
        assert inv.dependency_failure_is_explicit(
            succeeded=False, status=503, message="could not reach the predictor"
        ).held

    def test_succeeding_against_a_dead_dependency_violates_it(self) -> None:
        assert not inv.dependency_failure_is_explicit(succeeded=True, status=200, message="ok").held

    def test_failing_silently_violates_it(self) -> None:
        """An operation that fails without saying why is not an explicit failure."""
        assert not inv.dependency_failure_is_explicit(succeeded=False, status=None, message="").held


class TestTheEvidenceReport:
    def _evidence(self, **overrides: Any) -> ScenarioEvidence:
        defaults: dict[str, Any] = {
            "scenario": "example",
            "description": "d",
            "mode": MODE_LIVE,
            "failure_injected": "scaled something to zero",
            "expected_behaviour": "it failed closed",
        }
        return ScenarioEvidence(**{**defaults, **overrides})

    def test_a_scenario_only_passes_when_it_recovered(self) -> None:
        """Proving an invariant while leaving the cluster broken is not a pass."""
        evidence = self._evidence()
        evidence.add(inv.production_alias_unchanged("1", "1"))
        evidence.recovered = False
        assert evidence.held is True
        assert evidence.passed is False

    def test_a_scenario_with_no_invariants_does_not_pass(self) -> None:
        """A report that checked nothing proves nothing."""
        evidence = self._evidence()
        evidence.recovered = True
        assert evidence.passed is False

    def test_a_violated_invariant_fails_the_scenario(self) -> None:
        evidence = self._evidence()
        evidence.add(inv.production_alias_unchanged("1", "2"))
        evidence.recovered = True
        assert evidence.passed is False

    def test_an_error_fails_the_scenario(self) -> None:
        evidence = self._evidence()
        evidence.add(inv.production_alias_unchanged("1", "1"))
        evidence.recovered = True
        evidence.error = "kubectl exploded"
        assert evidence.passed is False

    def test_the_report_is_machine_readable(self) -> None:
        import json

        evidence = self._evidence()
        evidence.add(inv.production_alias_unchanged("1", "1"))
        evidence.recovered = True
        evidence.finish()

        report = FailureReport(run_id=new_run_id(), scenarios=[evidence])
        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["passed"] is True
        scenario = payload["scenarios"][0]
        for field in (
            "failure_injected",
            "expected_behaviour",
            "observed_behaviour",
            "blast_radius",
            "recovery_action",
            "recovery_result",
            "invariants",
        ):
            assert field in scenario, field
        assert payload["invariants_checked"] == [inv.PRODUCTION_ALIAS_UNCHANGED]

    def test_the_mode_distinguishes_live_from_a_double(self) -> None:
        """A reader is entitled to know which they are looking at."""
        assert MODE_LIVE != MODE_DOUBLE
        assert self._evidence(mode=MODE_DOUBLE).to_dict()["mode"] == MODE_DOUBLE

    def test_a_report_is_written_to_disk(self, tmp_path: Any) -> None:
        evidence = self._evidence()
        evidence.add(inv.production_alias_unchanged("1", "1"))
        evidence.recovered = True
        report = FailureReport(run_id="failure-test", scenarios=[evidence])
        written = report.write(tmp_path)
        assert written.exists()
        assert written.name == "failure-test.json"


class TestTheHarnessRestoresWhatItBreaks:
    """The property that makes it safe to run at all."""

    def test_live_scenarios_are_skipped_without_a_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ml_platform.failure import runner
        from ml_platform.failure.cluster import ClusterControl

        monkeypatch.setattr(ClusterControl, "available", lambda _self: False)
        report = runner.run(
            _StubConfig(), base_url="http://unused", names=["model_serving_failure"], live=True
        )
        assert report.scenarios == []

    def test_an_unknown_scenario_is_refused(self) -> None:
        from ml_platform.failure import runner

        with pytest.raises(ValueError, match="unknown scenario"):
            runner.run(_StubConfig(), base_url="http://unused", names=["nonsense"], live=False)

    def test_scale_records_what_to_restore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restoring to a remembered value, not to a hardcoded 1."""
        from ml_platform.failure.cluster import ClusterControl

        calls: list[tuple[str, ...]] = []

        def _fake(_self: Any, *args: str, check: bool = True) -> str:
            del check
            calls.append(args)
            return "3" if args[0] == "get" else ""

        monkeypatch.setattr(ClusterControl, "_kubectl", _fake)
        control = ClusterControl()
        state = control.scale("thing", 0)
        assert state.replicas == 3

        control.restore(state)
        assert ("scale", "deployment/thing", "--replicas=3") in calls

    def test_every_live_scenario_names_a_restore_action(self) -> None:
        """A scenario whose recovery_action is empty never said how to undo it."""
        import inspect

        from ml_platform.failure import scenarios

        for name in (
            "model_serving_failure",
            "mlflow_failure",
            "canary_failure",
            "telemetry_failure",
        ):
            source = inspect.getsource(getattr(scenarios, f"scenario_{name}"))
            assert "finally:" in source, name
            assert "recovery_action" in source, name


class _StubConfig:
    """Only what the runner touches before it decides to skip."""

    report_dir = "artifacts/reports"


class TestTheBadCandidateScenario:
    """The one scenario that uses a double, run end to end here.

    The gates, the comparison and the registration code are the real ones; only
    the data and the registry are isolated, because destroying the live registry
    to prove that a bad model is rejected would be worse evidence, not better.
    """

    def _config(self, synthetic_config: Any) -> Any:
        """The synthetic register plus the project's actual gate configuration."""
        import yaml

        from ml_platform.config import Config
        from ml_platform.paths import project_root

        raw = yaml.safe_load((project_root() / "configs" / "base.yaml").read_text(encoding="utf-8"))
        return Config(
            raw={
                **synthetic_config.raw,
                "promotion": raw["promotion"],
                "tracking": {"enabled": False},
            },
            environment=synthetic_config.environment,
        )

    def test_the_gates_reject_it_and_production_is_untouched(self, synthetic_config: Any) -> None:
        from ml_platform.failure.report import MODE_DOUBLE
        from ml_platform.failure.scenarios import scenario_bad_candidate

        evidence = scenario_bad_candidate(self._config(synthetic_config))

        assert evidence.error is None, evidence.error
        assert evidence.mode == MODE_DOUBLE, "a double must say so in its report"
        assert evidence.passed, evidence.summary()

        observations = evidence.observations
        assert observations["promote"] is False
        assert not observations["registered"]
        assert observations["failed_gates"], "a rejection must name the gates that failed"
        assert observations["alias_before"] == observations["alias_after"]

    def test_it_proves_the_intended_invariant(self, synthetic_config: Any) -> None:
        from ml_platform.failure.invariants import (
            FAILED_CANDIDATE_NOT_PRODUCTION,
            PRODUCTION_ALIAS_UNCHANGED,
        )
        from ml_platform.failure.scenarios import scenario_bad_candidate

        evidence = scenario_bad_candidate(self._config(synthetic_config))
        names = {i.name for i in evidence.invariants}
        assert FAILED_CANDIDATE_NOT_PRODUCTION in names
        assert PRODUCTION_ALIAS_UNCHANGED in names
        assert all(i.held for i in evidence.invariants)

    def test_its_blast_radius_is_recorded_as_nothing(self, synthetic_config: Any) -> None:
        """A rejected candidate costs a training run and nothing else."""
        from ml_platform.failure.scenarios import scenario_bad_candidate

        evidence = scenario_bad_candidate(self._config(synthetic_config))
        assert evidence.blast_radius
        assert "none" in evidence.blast_radius[0].lower()
        assert evidence.recovered is True
