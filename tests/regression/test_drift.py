"""Tests for drift detection and the retraining path it triggers.

The properties worth protecting here are not "does PSI compute correctly" but
the ones that decide whether the whole arrangement is honest:

* a detector that cannot say *no* is useless, so the no-drift control must hold;
* the same window must produce the same report, or a drift decision cannot be
  reviewed after the fact;
* a scenario must never touch an outcome column, because manufacturing a label
  to go with manufactured inputs is the one thing this milestone must not do;
* retraining must reach the M6 gates rather than the registry, and a retrained
  candidate that is worse must be rejected.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_platform.monitoring import drift
from ml_platform.monitoring.windows import (
    PROTECTED_COLUMNS,
    SCENARIOS,
    CurrentWindow,
    WindowError,
    apply_scenario,
    build_current_window,
    slice_window,
)

SEED = 20260912


def _register(n: int = 4000, seed: int = SEED, shift: float = 0.0) -> pd.DataFrame:
    """A register-shaped frame, optionally shifted."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2008-01-01")
    approval = start + pd.to_timedelta(np.sort(rng.integers(0, 540, size=n)), unit="D")
    gross = rng.lognormal(11.5 + shift, 0.8, size=n)
    return pd.DataFrame(
        {
            "LoanNr_ChkDgt": [str(1_000_000 + i) for i in range(n)],
            "ApprovalDate": approval,
            "DisbursementDate": approval + pd.to_timedelta(rng.integers(5, 120, size=n), unit="D"),
            "Term": rng.integers(60, 300, size=n).astype(float),
            "NoEmp": rng.integers(0, 40, size=n).astype(float),
            "CreateJob": rng.integers(0, 8, size=n).astype(float),
            "RetainedJob": rng.integers(0, 15, size=n).astype(float),
            "GrAppv": gross,
            "SBA_Appv": gross * rng.uniform(0.5, 0.9, size=n),
            "DisbursementGross": gross * rng.uniform(0.85, 1.0, size=n),
            "State": rng.choice(["CA", "TX", "NY", "FL"], size=n),
            "BankState": rng.choice(["CA", "TX", "NY", "FL"], size=n),
            "NAICS": rng.choice(["722410", "451120", "621210", "236220"], size=n),
            "RevLineCr": rng.choice(["Y", "N"], size=n),
            "LowDoc": rng.choice(["Y", "N"], size=n),
            "UrbanRural": rng.choice([0.0, 1.0, 2.0], size=n),
            "NewExist": rng.choice([1.0, 2.0], size=n),
            "FranchiseCode": rng.choice([0.0, 1.0], size=n),
            "MIS_Status": rng.choice(["P I F", "CHGOFF"], size=n),
            "ChgOffDate": pd.NaT,
            "ChgOffPrinGr": 0.0,
            "BalanceGross": 0.0,
            "target": rng.integers(0, 2, size=n),
        }
    )


class TestNoDrift:
    """A detector that always fires is as useless as one that never does."""

    def test_a_frame_against_itself_shows_nothing(self) -> None:
        frame = _register()
        report = drift.compare_frames(frame, frame.copy())
        assert report.n_drifted == 0
        assert report.max_psi == pytest.approx(0.0, abs=1e-9)
        assert report.drift_detected is False
        assert report.decision == "no_action"

    def test_two_samples_of_one_population_show_nothing(self) -> None:
        """Sampling noise must not be reported as drift."""
        frame = _register(n=8000)
        first, second = frame.iloc[::2], frame.iloc[1::2]
        report = drift.compare_frames(first, second)
        assert report.drift_detected is False, [f.describe() for f in report.drifted_features]

    def test_the_decision_never_asks_for_a_rollback(self) -> None:
        """Drift observes inputs. It has no authority over what is served."""
        for frame in (_register(), _register(shift=1.2)):
            report = drift.compare_frames(_register(), frame)
            assert report.decision in {"retrain", "no_action"}


class TestDriftIsDetected:
    def test_a_shifted_population_is_caught(self) -> None:
        report = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=1.0))
        assert report.drift_detected is True
        assert report.decision == "retrain"
        assert report.max_psi > drift.PSI_MINOR

    def test_the_drifted_features_are_named(self) -> None:
        """A report that says "something moved" is not actionable."""
        report = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=1.0))
        names = {f.feature for f in report.drifted_features}
        assert names
        assert any("gross" in n or "amount" in n or "GrAppv" in n for n in names)

    def test_severity_follows_the_conventional_bands(self) -> None:
        report = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=1.5))
        worst = max(report.features, key=lambda f: f.psi)
        assert worst.severity == "major"
        assert worst.psi >= drift.PSI_MAJOR

    def test_one_moving_feature_is_not_a_population_event(self) -> None:
        """Requiring several features is what stops a blip churning the registry."""
        frame = _register()
        current = frame.copy()
        current["NoEmp"] = current["NoEmp"] * 6.0
        report = drift.compare_frames(frame, current, min_drifted_features=3)
        assert report.n_drifted >= 1
        assert report.drift_detected is False
        assert report.decision == "no_action"

    def test_unseen_categories_are_drift(self) -> None:
        frame = _register()
        current = frame.copy()
        current.loc[current.index[:1500], "NAICS"] = "999990"
        current.loc[current.index[:1500], "State"] = "NV"
        report = drift.compare_frames(frame, current, min_drifted_features=1)
        assert report.drift_detected is True

    def test_the_report_is_machine_readable(self) -> None:
        import json

        report = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=1.0))
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["decision"] in {"retrain", "no_action"}
        assert payload["thresholds"]["psi"]
        assert len(payload["features"]) == payload["n_features"]
        for feature in payload["features"]:
            assert {"feature", "psi", "drifted", "severity"} <= set(feature)


class TestReportsAreReproducible:
    """A drift decision that cannot be reproduced cannot be reviewed."""

    def test_the_same_inputs_give_the_same_report(self) -> None:
        first = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=0.8))
        second = drift.compare_frames(_register(), _register(seed=SEED + 1, shift=0.8))
        assert first.to_dict() == second.to_dict()

    def test_a_scenario_is_a_pure_function_of_its_seed(self) -> None:
        frame = _register()
        first = apply_scenario(frame, "lending_shift", seed=7)
        second = apply_scenario(frame, "lending_shift", seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_give_different_windows(self) -> None:
        frame = _register()
        assert not apply_scenario(frame, "lending_shift", 7).equals(
            apply_scenario(frame, "lending_shift", 8)
        )

    def test_the_window_fingerprint_is_stable(self) -> None:
        frame = _register()
        window = CurrentWindow(frame, date(2008, 1, 1), date(2009, 6, 30), "none", SEED)
        again = CurrentWindow(frame.copy(), date(2008, 1, 1), date(2009, 6, 30), "none", SEED)
        assert window.fingerprint() == again.fingerprint()

    def test_a_changed_row_changes_the_fingerprint(self) -> None:
        frame = _register()
        altered = frame.copy()
        altered.loc[0, "GrAppv"] = altered.loc[0, "GrAppv"] * 2
        base = CurrentWindow(frame, date(2008, 1, 1), date(2009, 6, 30), "none", SEED)
        moved = CurrentWindow(altered, date(2008, 1, 1), date(2009, 6, 30), "none", SEED)
        assert base.fingerprint() != moved.fingerprint()

    def test_slicing_is_deterministic(self) -> None:
        frame = _register()
        first = slice_window(frame, date(2008, 3, 1), date(2008, 9, 30), "ApprovalDate")
        second = slice_window(frame, date(2008, 3, 1), date(2008, 9, 30), "ApprovalDate")
        pd.testing.assert_frame_equal(first, second)
        assert 0 < len(first) < len(frame)


class TestScenariosNeverFabricateGroundTruth:
    """The rule this milestone must not break."""

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_no_outcome_column_is_touched(self, scenario: str) -> None:
        frame = _register()
        result = apply_scenario(frame, scenario, SEED)
        for column in PROTECTED_COLUMNS & set(frame.columns):
            pd.testing.assert_series_equal(
                frame[column], result[column], check_names=False, obj=column
            )

    def test_a_scenario_that_touched_an_outcome_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard, not merely the convention."""

        def _cheat(frame: pd.DataFrame, rng: Any) -> pd.DataFrame:
            del rng
            out = frame.copy()
            out["target"] = 0
            return out

        monkeypatch.setitem(SCENARIOS, "cheating", _cheat)
        with pytest.raises(WindowError, match="modified 'target'"):
            apply_scenario(_register(), "cheating", SEED)

    def test_an_unknown_scenario_is_refused(self) -> None:
        with pytest.raises(WindowError, match="unknown drift scenario"):
            apply_scenario(_register(), "wishful-thinking", SEED)

    def test_the_control_scenario_changes_nothing(self) -> None:
        frame = _register()
        pd.testing.assert_frame_equal(frame, apply_scenario(frame, "none", SEED))

    def test_row_count_is_never_changed(self) -> None:
        """A scenario shifts a distribution; it does not add or drop loans."""
        frame = _register()
        for scenario in SCENARIOS:
            assert len(apply_scenario(frame, scenario, SEED)) == len(frame)


class TestLabelMaturityIsEnforced:
    """Retraining may only use outcomes that had actually resolved."""

    def test_a_window_past_the_cutoff_is_refused(self) -> None:
        from ml_platform.config import load_config
        from ml_platform.pipelines.retrain_pipeline import MaturityError, assert_labels_matured

        config = load_config("production")
        with pytest.raises(MaturityError, match="label maturity cutoff"):
            assert_labels_matured(config, date(2013, 1, 1))

    def test_the_configured_window_is_within_the_cutoff(self) -> None:
        from ml_platform.config import load_config
        from ml_platform.pipelines.retrain_pipeline import assert_labels_matured

        config = load_config("production")
        assert_labels_matured(config, config.drift_window_end)

    def test_the_cutoff_is_the_horizon_before_observation_end(self) -> None:
        """Not an arbitrary date: it is when a 60-month horizon could have closed."""
        from ml_platform.config import load_config

        config = load_config("production")
        assert config.label_maturity_end.year == config.observation_end.year - 5


class TestTheWindowBuilder:
    def test_it_uses_the_configured_window(self, synthetic_config: Any) -> None:
        frame = _register()
        window = build_current_window(
            frame, synthetic_config, start=date(2008, 1, 1), end=date(2008, 6, 30), scenario="none"
        )
        assert window.n_rows > 0
        assert window.start == date(2008, 1, 1)
        assert window.scenario == "none"

    def test_an_empty_window_is_an_error_not_an_empty_report(self, synthetic_config: Any) -> None:
        """Silently comparing against nothing would report no drift forever."""
        with pytest.raises(WindowError, match="no rows"):
            build_current_window(
                _register(), synthetic_config, start=date(1990, 1, 1), end=date(1990, 12, 31)
            )

    def test_it_describes_itself_for_the_report(self, synthetic_config: Any) -> None:
        window = build_current_window(
            _register(), synthetic_config, start=date(2008, 1, 1), end=date(2009, 6, 30)
        )
        described = window.describe()
        assert {"start", "end", "scenario", "seed", "n_rows", "fingerprint"} <= set(described)
