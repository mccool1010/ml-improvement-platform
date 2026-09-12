"""Tests for the drift -> retrain -> gates -> registry path.

The thing being protected is the shape of the flow, not the accuracy of any
model. Retraining must produce a *candidate* that the existing M6 gates judge,
and a retrained model that is worse must be rejected with production left alone.
A retraining pipeline that could promote its own output would make every gate in
the project decorative.

These run on the synthetic register from conftest, so they train in seconds.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from ml_platform.config import Config
from ml_platform.data.splitting import make_splits
from ml_platform.monitoring.windows import CurrentWindow
from ml_platform.pipelines.drift_pipeline import DriftEvent, run_drift_check
from ml_platform.pipelines.retrain_pipeline import RetrainOutcome, run_retraining
from ml_platform.pipelines.train_pipeline import load_prepared_dataset, run_training


def _real_promotion_block() -> dict[str, Any]:
    """The project's actual gate configuration, not a relaxed copy.

    Loaded from configs/base.yaml so these tests exercise the seven gates that
    guard production. A test fixture with easier gates would prove that
    retraining reaches *a* gate, which is not the property that matters.
    """
    import yaml

    from ml_platform.paths import project_root

    raw = yaml.safe_load((project_root() / "configs" / "base.yaml").read_text(encoding="utf-8"))
    promotion: dict[str, Any] = raw["promotion"]
    return promotion


def _drift_config(base: Config, **overrides: Any) -> Config:
    """The synthetic config with a monitoring block pointed at its own dates."""
    monitoring = {
        "label_maturity_end": "2009-06-30",
        "drift": {
            "reference_split": "train",
            "feature_set": "engineered",
            "threshold_psi": 0.1,
            "min_drifted_features": 3,
            "window": {"start": "2006-01-01", "end": "2009-06-30", "scenario": "none"},
            **overrides,
        },
    }
    return Config(
        raw={
            **base.raw,
            "monitoring": monitoring,
            "promotion": _real_promotion_block(),
            "tracking": {"enabled": False},
        },
        environment=base.environment,
    )


@pytest.fixture
def drift_config(synthetic_config: Config) -> Config:
    return _drift_config(synthetic_config)


class TestTheDriftCheckRuns:
    def test_it_produces_a_report_and_an_event_id(self, drift_config: Config) -> None:
        event = run_drift_check(config=drift_config)
        assert event.event_id.startswith("drift-")
        assert event.decision in {"retrain", "no_action"}
        assert event.report.features
        assert event.report_path.endswith(".json")

    def test_the_report_is_written_to_disk(self, drift_config: Config) -> None:
        import json
        from pathlib import Path

        event = run_drift_check(config=drift_config)
        payload = json.loads(Path(event.report_path).read_text(encoding="utf-8"))
        assert payload["drift_event_id"] == event.event_id
        assert payload["window"]["fingerprint"]
        assert payload["features"]

    def test_the_scenario_changes_the_verdict(self, drift_config: Config) -> None:
        """The control and the drifted scenario must not agree, or the whole
        demonstration proves nothing."""
        quiet = run_drift_check(config=drift_config, scenario="none")
        loud = run_drift_check(config=drift_config, scenario="lending_shift")
        assert loud.report.max_psi > quiet.report.max_psi
        assert loud.report.n_drifted >= quiet.report.n_drifted


class TestNoDriftMeansNoRetraining:
    def test_production_is_left_alone(self, drift_config: Config) -> None:
        """The cheapest correct behaviour: do nothing, and say so."""
        event = run_drift_check(config=drift_config, scenario="none")
        # Force the verdict to no-drift regardless of what the synthetic data
        # happens to show, so this test is about the branch, not the fixture.
        event.report.min_drifted_features = 10_000

        outcome = run_retraining(config=drift_config, drift_event=event, register=False)
        assert outcome.triggered is False
        assert outcome.decision is None
        assert outcome.candidate_run_id is None
        assert "no_action" in outcome.reason

    def test_force_retrains_anyway(self, drift_config: Config) -> None:
        """A scheduled refresh is a legitimate trigger; it changes nothing about
        how the result is judged."""
        event = run_drift_check(config=drift_config, scenario="none")
        event.report.min_drifted_features = 10_000

        outcome = run_retraining(config=drift_config, drift_event=event, force=True, register=False)
        assert outcome.triggered is True
        assert outcome.reason == "forced by request"
        assert outcome.decision is not None


class TestRetrainingReachesTheGates:
    def test_a_candidate_is_judged_not_installed(self, drift_config: Config) -> None:
        outcome = run_retraining(
            config=drift_config, scenario="lending_shift", force=True, register=False
        )
        assert outcome.triggered is True
        assert outcome.decision is not None
        # The seven M6 gates, unchanged and applied to the retrained candidate.
        assert len(outcome.decision.report.gates) == len(drift_config.gate_config)
        assert outcome.decision.report.decision_split == drift_config.decision_split

    def test_the_decision_is_made_on_validation_never_test(self, drift_config: Config) -> None:
        """The held-out split stays held out, retraining or not."""
        outcome = run_retraining(
            config=drift_config, scenario="lending_shift", force=True, register=False
        )
        assert outcome.decision is not None
        assert outcome.decision.report.decision_split == "validation"

    def test_the_window_is_added_to_training_only(self, drift_config: Config) -> None:
        """Evaluation splits must be identical to a normal run, or the gate is
        comparing against metrics measured on different rows."""
        prepared, _ = load_prepared_dataset(drift_config)
        splits = make_splits(prepared, drift_config)
        window = prepared[pd.to_datetime(prepared["ApprovalDate"]) >= pd.Timestamp("2006-01-01")]

        plain = run_training(config=drift_config, model_key="candidate", save_model=False)
        retrained = run_training(
            config=drift_config,
            model_key="candidate",
            save_model=False,
            extra_training_frame=window,
            training_note="test",
        )

        def _rows(record: Any, name: str) -> int:
            return next(s["n_rows"] for s in record.splits if s["name"] == name)

        for name in ("validation", "test"):
            assert _rows(plain, name) == _rows(retrained, name) == splits[name].n_rows

        extra = next(s for s in retrained.splits if s["name"] == "retraining_window")
        assert extra["n_rows"] == len(window)
        assert extra["used_for_training"] is True

    def test_retraining_actually_changes_the_model(self, drift_config: Config) -> None:
        """If the extra rows made no difference, nothing was retrained."""
        plain = run_training(config=drift_config, model_key="candidate", save_model=False)
        prepared, _ = load_prepared_dataset(drift_config)
        window = prepared[pd.to_datetime(prepared["ApprovalDate"]) >= pd.Timestamp("2006-01-01")]
        retrained = run_training(
            config=drift_config,
            model_key="candidate",
            save_model=False,
            extra_training_frame=window,
        )
        assert (
            plain.metrics["validation"]["average_precision"]
            != retrained.metrics["validation"]["average_precision"]
        )


class TestAWorseCandidateIsRejected:
    """The property the whole milestone rests on."""

    def test_a_deliberately_crippled_retrain_does_not_get_promoted(
        self, drift_config: Config
    ) -> None:
        # A single-iteration, near-zero-learning-rate model is far worse than the
        # bootstrap incumbent, so the gates must refuse it.
        outcome = run_retraining(
            config=drift_config,
            scenario="lending_shift",
            force=True,
            register=True,
            param_overrides={"max_iter": 1, "learning_rate": 0.0001, "max_depth": 1},
        )
        assert outcome.triggered is True
        assert outcome.decision is not None
        assert outcome.promoted is False, outcome.decision.report.summary()
        assert outcome.registered is False
        assert outcome.decision.report.failures
        assert any("production is unchanged" in note for note in outcome.notes)

    def test_rejection_names_the_gates_that_failed(self, drift_config: Config) -> None:
        outcome = run_retraining(
            config=drift_config,
            scenario="lending_shift",
            force=True,
            register=False,
            param_overrides={"max_iter": 1, "learning_rate": 0.0001, "max_depth": 1},
        )
        assert outcome.decision is not None
        names = {gate.name for gate in outcome.decision.report.failures}
        assert names, "a rejected candidate must say which gates it failed"
        for gate in outcome.decision.report.failures:
            assert gate.reason


class TestTheOutcomeIsTraceable:
    def test_the_outcome_records_the_drift_event(self, drift_config: Config) -> None:
        outcome = run_retraining(
            config=drift_config, scenario="lending_shift", force=True, register=False
        )
        payload = outcome.to_dict()
        assert payload["drift_event_id"] == outcome.drift_event.event_id
        assert payload["window"]["scenario"] == "lending_shift"
        assert payload["window"]["fingerprint"]
        assert payload["candidate_run_id"]

    def test_the_report_is_written(self, drift_config: Config) -> None:
        from pathlib import Path

        outcome = run_retraining(
            config=drift_config, scenario="lending_shift", force=True, register=False
        )
        assert outcome.report_path is not None
        assert Path(outcome.report_path).exists()

    def test_an_untriggered_run_still_reports(self, drift_config: Config) -> None:
        event = run_drift_check(config=drift_config, scenario="none")
        event.report.min_drifted_features = 10_000
        outcome = run_retraining(config=drift_config, drift_event=event, register=False)
        assert outcome.to_dict()["triggered"] is False


class TestTheEvaluationSplitsAreProtected:
    def test_a_window_overlapping_validation_is_refused(self, drift_config: Config) -> None:
        """Training on evaluation rows would make every gate meaningless, so it
        fails loudly rather than producing a wonderful candidate."""
        from ml_platform.pipelines.retrain_pipeline import _assert_disjoint_from_evaluation

        prepared, _ = load_prepared_dataset(drift_config)
        splits = make_splits(prepared, drift_config)
        leaking = splits["validation"].frame

        with pytest.raises(ValueError, match="overlaps the validation split"):
            _assert_disjoint_from_evaluation(leaking, splits, drift_config)

    def test_the_real_window_is_disjoint(self, drift_config: Config) -> None:
        from ml_platform.pipelines.retrain_pipeline import _assert_disjoint_from_evaluation

        prepared, _ = load_prepared_dataset(drift_config)
        splits = make_splits(prepared, drift_config)
        window = CurrentWindow(
            frame=prepared[pd.to_datetime(prepared["ApprovalDate"]) >= pd.Timestamp("2006-01-01")],
            start=date(2006, 1, 1),
            end=date(2009, 6, 30),
            scenario="none",
            seed=42,
        )
        _assert_disjoint_from_evaluation(window.frame, splits, drift_config)


def test_a_drift_event_round_trips_through_its_dict(drift_config: Config) -> None:
    event: DriftEvent = run_drift_check(config=drift_config, scenario="lending_shift")
    payload = event.to_dict()
    assert payload["drift_event_id"] == event.event_id
    assert payload["report"]["decision"] == event.decision
    assert isinstance(RetrainOutcome(event, False, "x").to_dict(), dict)
