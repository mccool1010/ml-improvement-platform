"""Regression protection for the two M0 labelling findings.

Both findings were discovered by measurement, and both produced results that
looked good. These tests exist so that a future refactor cannot quietly
reintroduce either one.

**Finding 1: the label must be a fixed horizon, not eventual outcome.**
Filtering to loans that have resolved is biased, because defaults resolve in a
median 1,410 days while healthy loans resolve only at scheduled maturity. The
resolution filter retained 85% of 2006 approvals but 26% of 2010, and reported a
69% default rate for 2008, which is not credible.

**Finding 2: the population must have uniform exposure.**
A fixed horizon alone still lets the target measure whether a loan survived long
enough to default. Without the restriction, the default rate cliffed from 41.0%
for three-to-five year loans to 8.3% for five-to-seven year loans, exactly at the
60-month boundary, and ``Term`` alone reproduced 96% of full-model performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.data import preprocessing as pp

OBSERVATION_END = pd.Timestamp("2014-06-25")
HORIZON = 60


def _loan(
    *,
    status: str,
    disbursed: str,
    term: int,
    chargeoff: str | None = None,
) -> dict[str, object]:
    return {
        "MIS_Status": status,
        "DisbursementDate": pd.Timestamp(disbursed),
        "ChgOffDate": pd.Timestamp(chargeoff) if chargeoff else pd.NaT,
        "Term": term,
    }


class TestFixedHorizonIsNotEventualOutcome:
    """The label must ask "within the horizon", never "ever"."""

    def test_a_late_chargeoff_is_negative_not_positive(self) -> None:
        """The defining difference between the two label definitions.

        This loan did eventually charge off. Under the rejected definition it
        would be positive. Under the fixed horizon it is negative, because the
        charge-off happened in month 96.
        """
        frame = pd.DataFrame(
            [_loan(status="CHGOFF", disbursed="2000-01-01", term=120, chargeoff="2008-01-01")]
        )
        result = pp.build_label(frame, HORIZON, OBSERVATION_END)
        assert result["target"].iloc[0] == 0

    def test_horizon_boundary_is_inclusive_and_stable(self) -> None:
        """A charge-off at the horizon edge is positive; just past it is not."""
        inside = pd.Timestamp("2000-01-01") + pd.to_timedelta(HORIZON * 30.44 - 1, unit="D")
        outside = pd.Timestamp("2000-01-01") + pd.to_timedelta(HORIZON * 30.44 + 1, unit="D")
        frame = pd.DataFrame(
            [
                _loan(
                    status="CHGOFF", disbursed="2000-01-01", term=120, chargeoff=str(inside.date())
                ),
                _loan(
                    status="CHGOFF", disbursed="2000-01-01", term=120, chargeoff=str(outside.date())
                ),
            ]
        )
        result = pp.build_label(frame, HORIZON, OBSERVATION_END)
        assert result["target"].tolist() == [1, 0]

    def test_observability_does_not_depend_on_the_outcome(self) -> None:
        """The bias guard.

        Two loans disbursed on the same day, one defaulting early and one healthy,
        must be included or excluded together. If observability ever consults the
        outcome, defaults survive filtering more often than healthy loans and the
        measured default rate inflates.
        """
        frame = pd.DataFrame(
            [
                _loan(status="CHGOFF", disbursed="2005-01-01", term=120, chargeoff="2006-01-01"),
                _loan(status="P I F", disbursed="2005-01-01", term=120),
            ]
        )
        result = pp.build_label(frame, HORIZON, OBSERVATION_END)
        assert result["observable"].nunique() == 1
        assert result["horizon_end"].nunique() == 1

    def test_resolution_speed_does_not_change_inclusion(self) -> None:
        """Charge-offs at wildly different speeds are all equally observable."""
        frame = pd.DataFrame(
            [
                _loan(status="CHGOFF", disbursed="2005-01-01", term=240, chargeoff=d)
                for d in ("2005-06-01", "2007-01-01", "2009-06-01")
            ]
        )
        result = pp.build_label(frame, HORIZON, OBSERVATION_END)
        assert result["observable"].all()

    def test_unresolved_loans_are_excluded(self) -> None:
        """A loan with no recorded outcome has no label and cannot be used."""
        frame = pd.DataFrame([_loan(status="", disbursed="2000-01-01", term=120)])
        result = pp.build_label(frame, HORIZON, OBSERVATION_END)
        assert not result["observable"].iloc[0]

    def test_horizon_is_configurable_and_changes_the_label(self) -> None:
        """The horizon is a parameter, not a constant baked into the logic."""
        frame = pd.DataFrame(
            [_loan(status="CHGOFF", disbursed="2000-01-01", term=240, chargeoff="2006-01-01")]
        )
        assert pp.build_label(frame, 60, OBSERVATION_END)["target"].iloc[0] == 0
        assert pp.build_label(frame, 84, OBSERVATION_END)["target"].iloc[0] == 1


class TestUniformExposure:
    """The population must not let the model read the observation boundary."""

    def test_loans_shorter_than_the_horizon_are_excluded(self) -> None:
        frame = pd.DataFrame({"Term": [0, 12, 36, 59, 60, 84, 240, 569]})
        kept = pp.restrict_to_uniform_exposure(frame, HORIZON)
        assert kept["Term"].tolist() == [60, 84, 240, 569]

    def test_every_retained_loan_survives_the_whole_horizon(self) -> None:
        """The invariant itself: no retained loan matures before the horizon ends."""
        frame = pd.DataFrame({"Term": np.arange(0, 600, 3)})
        kept = pp.restrict_to_uniform_exposure(frame, HORIZON)
        assert (kept["Term"] >= HORIZON).all()

    def test_prepare_applies_the_restriction_when_configured(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    **_loan(status="P I F", disbursed="2000-01-01", term=t),
                    "ApprovalDate": pd.Timestamp("2000-01-01"),
                }
                for t in (24, 36, 84, 240)
            ]
        )
        labelled = pp.build_label(raw, HORIZON, OBSERVATION_END)
        kept = pp.restrict_to_uniform_exposure(labelled, HORIZON)
        assert sorted(kept["Term"].tolist()) == [84, 240]

    def test_restriction_is_a_no_op_when_all_terms_are_long(self) -> None:
        frame = pd.DataFrame({"Term": [60, 120, 240]})
        assert len(pp.restrict_to_uniform_exposure(frame, HORIZON)) == 3

    @pytest.mark.parametrize("horizon", [24, 36, 60, 84])
    def test_restriction_tracks_the_configured_horizon(self, horizon: int) -> None:
        """Horizon and minimum term are coupled.

        If the horizon changes and the minimum term does not, exposure stops
        being uniform and the boundary artifact returns.
        """
        frame = pd.DataFrame({"Term": np.arange(0, 300, 6)})
        kept = pp.restrict_to_uniform_exposure(frame, horizon)
        assert (kept["Term"] >= horizon).all()
        assert len(kept) == int((frame["Term"] >= horizon).sum())
