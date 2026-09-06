"""Tests for the promotion quality gates.

These are the tests that protect production. Every one of them is a way a worse
model could reach production if the gate logic were wrong, so the failure paths
get more attention here than the happy path.

The gates are pure functions over two metric dictionaries, so the whole file
runs in milliseconds and no model is ever fitted.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform.promotion import gates

# Real validation numbers from configs/reference.yaml, so the fixtures behave
# like the models this project actually produces.
PRODUCTION: dict[str, Any] = {
    "average_precision": 0.131498,
    "roc_auc": 0.753873,
    "brier_score": 0.043644,
    "brier_skill_score": 0.029828,
    "recall_at_capacity": 0.338275,
    "inference_seconds": 0.35,
    "n_rows": 54284,
}
CANDIDATE: dict[str, Any] = {
    "average_precision": 0.732031,
    "roc_auc": 0.973004,
    "brier_score": 0.022446,
    "brier_skill_score": 0.501043,
    "recall_at_capacity": 0.868513,
    "inference_seconds": 0.83,
    "n_rows": 54284,
}

GATE_CONFIG: dict[str, Any] = {
    "min_improvement": {"metric": "average_precision", "min_delta": 0.01, "required": True},
    "minimum_metric": {"metric": "average_precision", "minimum": 0.25, "required": True},
    "roc_auc_regression": {"metric": "roc_auc", "max_decline": 0.005, "required": True},
    "calibration": {"max_decline": 0.0, "min_skill": 0.0, "required": True},
    "recall_regression": {"metric": "recall_at_capacity", "max_decline": 0.0, "required": True},
    "latency": {"max_mean_prediction_ms": 1.0, "required": True},
    "reproducibility": {"require_clean_revision": True, "required": True},
}

CLEAN_CONTEXT: dict[str, Any] = {
    "reproducible": True,
    "git_revision": "abc123",
    "git_dirty": False,
    "lockfile_sha256": "beef",
}


def _run(
    candidate: dict[str, Any] | None = None,
    production: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> gates.GateReport:
    return gates.evaluate_gates(
        {**CANDIDATE, **(candidate or {})},
        {**PRODUCTION, **(production or {})},
        config or GATE_CONFIG,
        context=context or CLEAN_CONTEXT,
    )


def _gate(report: gates.GateReport, name: str) -> gates.GateResult:
    return next(g for g in report.gates if g.name == name)


class TestTheHeldOutSplitIsProtected:
    """The rule the whole milestone rests on."""

    def test_deciding_on_the_test_split_is_refused(self) -> None:
        with pytest.raises(gates.GateConfigurationError, match="held-out evidence"):
            gates.evaluate_gates(CANDIDATE, PRODUCTION, GATE_CONFIG, split="test")

    def test_validation_is_accepted(self) -> None:
        assert _run().decision_split == "validation"

    def test_no_gates_configured_is_refused(self) -> None:
        """An empty gate set would promote everything silently."""
        with pytest.raises(gates.GateConfigurationError, match="no gates"):
            gates.evaluate_gates(CANDIDATE, PRODUCTION, {})


class TestAcceptedCandidate:
    def test_a_clearly_better_candidate_passes_every_gate(self) -> None:
        report = _run()
        assert report.promote
        assert report.failures == []
        assert all(g.passed for g in report.gates)

    def test_every_configured_gate_is_evaluated(self) -> None:
        assert {g.name for g in _run().gates} == set(GATE_CONFIG)

    def test_the_summary_states_the_verdict(self) -> None:
        assert "[PROMOTE]" in _run().summary()


class TestImprovementGate:
    def test_an_equal_candidate_is_rejected(self) -> None:
        """Matching production is not a reason to replace it."""
        report = _run(candidate={"average_precision": PRODUCTION["average_precision"]})
        assert not report.promote
        assert _gate(report, "min_improvement").blocking

    def test_an_improvement_below_the_margin_is_rejected(self) -> None:
        report = _run(candidate={"average_precision": PRODUCTION["average_precision"] + 0.005})
        gate = _gate(report, "min_improvement")
        assert not gate.passed
        assert gate.observed == pytest.approx(0.005, abs=1e-6)
        assert gate.threshold == 0.01

    def test_an_improvement_exactly_at_the_margin_passes(self) -> None:
        report = _run(candidate={"average_precision": PRODUCTION["average_precision"] + 0.01})
        assert _gate(report, "min_improvement").passed

    def test_a_worse_candidate_is_rejected(self) -> None:
        report = _run(candidate={"average_precision": 0.05})
        assert not report.promote
        assert _gate(report, "min_improvement").observed < 0

    def test_the_failure_explains_itself(self) -> None:
        report = _run(candidate={"average_precision": PRODUCTION["average_precision"]})
        assert "below the required" in _gate(report, "min_improvement").reason


class TestMinimumMetricGate:
    def test_a_candidate_below_the_floor_is_rejected(self) -> None:
        """Beating a poor incumbent is not enough on its own."""
        report = _run(candidate={"average_precision": 0.20}, production={"average_precision": 0.05})
        gate = _gate(report, "minimum_metric")
        assert not gate.passed
        assert gate.threshold == 0.25
        # The improvement gate is satisfied; only the floor stops it.
        assert _gate(report, "min_improvement").passed

    def test_a_candidate_at_the_floor_passes(self) -> None:
        report = _run(candidate={"average_precision": 0.25}, production={"average_precision": 0.05})
        assert _gate(report, "minimum_metric").passed

    def test_the_floor_is_configurable(self) -> None:
        config = {**GATE_CONFIG, "minimum_metric": {"metric": "average_precision", "minimum": 0.9}}
        assert not _gate(_run(config=config), "minimum_metric").passed


class TestRegressionGates:
    def test_a_roc_auc_regression_beyond_tolerance_is_rejected(self) -> None:
        """Guards against trading general ranking for a narrow gain."""
        report = _run(candidate={"roc_auc": PRODUCTION["roc_auc"] - 0.02})
        assert not report.promote
        assert _gate(report, "roc_auc_regression").blocking

    def test_a_roc_auc_regression_inside_tolerance_passes(self) -> None:
        report = _run(candidate={"roc_auc": PRODUCTION["roc_auc"] - 0.004})
        assert _gate(report, "roc_auc_regression").passed

    def test_a_recall_regression_is_rejected(self) -> None:
        report = _run(candidate={"recall_at_capacity": PRODUCTION["recall_at_capacity"] - 0.01})
        assert _gate(report, "recall_regression").blocking

    def test_lower_is_better_metrics_invert_correctly(self) -> None:
        """A smaller Brier score is an improvement, not a regression."""
        report = _run(candidate={"brier_score": 0.001})
        assert _gate(report, "calibration").passed


class TestCalibrationGate:
    def test_a_worse_brier_score_is_rejected(self) -> None:
        report = _run(candidate={"brier_score": PRODUCTION["brier_score"] + 0.01})
        gate = _gate(report, "calibration")
        assert not gate.passed
        assert "worse than production" in gate.reason

    def test_a_model_no_better_than_the_base_rate_is_rejected(self) -> None:
        """This is the check that caught balanced class weights back at M1."""
        report = _run(candidate={"brier_skill_score": -0.05})
        gate = _gate(report, "calibration")
        assert not gate.passed
        assert "no better calibrated" in gate.reason

    def test_both_calibration_failures_are_reported_together(self) -> None:
        report = _run(
            candidate={"brier_score": PRODUCTION["brier_score"] + 0.01, "brier_skill_score": -0.1}
        )
        reason = _gate(report, "calibration").reason
        assert "worse than production" in reason
        assert "no better calibrated" in reason


class TestLatencyGate:
    def test_a_fast_candidate_passes(self) -> None:
        gate = _gate(_run(), "latency")
        assert gate.passed
        assert gate.observed is not None and gate.observed < 1.0

    def test_a_slow_candidate_is_rejected(self) -> None:
        """An accurate model that cannot be served fast enough cannot be used."""
        report = _run(candidate={"inference_seconds": 600.0})
        gate = _gate(report, "latency")
        assert not gate.passed
        assert "exceeds the budget" in gate.reason

    def test_the_measurement_is_per_prediction(self) -> None:
        assert gates.mean_prediction_ms(
            {"inference_seconds": 1.0, "n_rows": 1000}
        ) == pytest.approx(1.0)

    def test_a_missing_measurement_fails_rather_than_passes(self) -> None:
        """An unmeasurable constraint must not be silently treated as satisfied."""
        report = _run(candidate={"inference_seconds": None, "n_rows": 0})
        gate = _gate(report, "latency")
        assert not gate.passed
        assert "cannot be checked" in gate.reason


class TestReproducibilityGate:
    def test_a_clean_run_passes(self) -> None:
        assert _gate(_run(), "reproducibility").passed

    def test_a_dirty_working_tree_is_rejected(self) -> None:
        report = _run(context={**CLEAN_CONTEXT, "reproducible": False, "git_dirty": True})
        gate = _gate(report, "reproducibility")
        assert not gate.passed
        assert "uncommitted changes" in gate.reason

    def test_an_unknown_revision_is_rejected(self) -> None:
        report = _run(context={**CLEAN_CONTEXT, "reproducible": False, "git_revision": "unknown"})
        assert "code version is unknown" in _gate(report, "reproducibility").reason

    def test_a_missing_lockfile_is_rejected(self) -> None:
        report = _run(context={**CLEAN_CONTEXT, "reproducible": False, "lockfile_sha256": "absent"})
        assert "lockfile" in _gate(report, "reproducibility").reason

    def test_the_check_can_be_relaxed_by_configuration(self) -> None:
        config = {**GATE_CONFIG, "reproducibility": {"require_clean_revision": False}}
        report = _run(config=config, context={**CLEAN_CONTEXT, "reproducible": False})
        assert _gate(report, "reproducibility").passed


class TestMultipleFailures:
    def test_every_failure_is_reported_not_just_the_first(self) -> None:
        report = _run(
            candidate={
                "average_precision": 0.10,
                "roc_auc": 0.50,
                "brier_score": 0.20,
                "brier_skill_score": -0.4,
                "recall_at_capacity": 0.10,
                "inference_seconds": 900.0,
            },
            context={**CLEAN_CONTEXT, "reproducible": False, "git_dirty": True},
        )
        assert not report.promote
        failed = {g.name for g in report.failures}
        assert failed == set(GATE_CONFIG), "every gate should have failed and been reported"

    def test_the_summary_counts_the_blocking_failures(self) -> None:
        report = _run(candidate={"average_precision": 0.05, "roc_auc": 0.4})
        assert "[REJECT]" in report.summary()
        assert len(report.failures) >= 2

    def test_one_failure_is_enough_to_reject(self) -> None:
        """Any mandatory gate failing must stop the promotion."""
        report = _run(candidate={"recall_at_capacity": 0.0})
        assert not report.promote
        assert len(report.failures) == 1


class TestOptionalGates:
    def test_a_non_required_failure_warns_without_blocking(self) -> None:
        config = {
            **GATE_CONFIG,
            "latency": {"max_mean_prediction_ms": 0.000001, "required": False},
        }
        report = _run(config=config)
        assert report.promote
        assert [g.name for g in report.warnings] == ["latency"]

    def test_a_non_required_gate_is_still_reported(self) -> None:
        config = {
            **GATE_CONFIG,
            "latency": {"max_mean_prediction_ms": 0.000001, "required": False},
        }
        assert "WARN" in _gate(_run(config=config), "latency").describe()


class TestReporting:
    def test_the_report_serialises_with_every_gate(self) -> None:
        payload = _run().to_dict()
        assert payload["promote"] is True
        assert payload["n_gates"] == len(GATE_CONFIG)
        assert payload["blocking_failures"] == []

    def test_a_rejection_names_the_blocking_gates(self) -> None:
        payload = _run(candidate={"average_precision": 0.05}).to_dict()
        assert payload["promote"] is False
        assert "min_improvement" in payload["blocking_failures"]

    def test_each_gate_records_observed_and_threshold(self) -> None:
        for gate in _run().gates:
            assert gate.comparison
            if gate.name != "reproducibility":
                assert gate.observed is not None

    def test_a_missing_metric_is_reported_clearly(self) -> None:
        with pytest.raises(gates.GateConfigurationError, match="no 'average_precision'"):
            gates.evaluate_gates({}, PRODUCTION, GATE_CONFIG, context=CLEAN_CONTEXT)
