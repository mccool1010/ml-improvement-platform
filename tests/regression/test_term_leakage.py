"""Regression protection: Term must not encode the observation window.

This is the subtlest of the three M0 findings and the easiest to reintroduce,
because reintroducing it *improves* every headline metric.

The label asks whether a loan charged off within 60 months of disbursement. If
the population contains loans that mature before 60 months, the target partly
measures whether the loan was still alive long enough to default at all. A
36-month loan lives its whole life inside the window; a 240-month loan is a
quarter of the way through.

Measured on the real dataset before the fix, the default rate cliffed from 41.0%
for three-to-five year loans to 8.3% for five-to-seven year loans, exactly at the
horizon boundary, and ``Term`` alone reproduced 96% of full-model performance
(0.7211 average precision against 0.7535 for every feature together). The model
was largely learning the shape of the observation window.

The fix restricts the population to ``Term >= horizon``. These tests assert the
resulting invariants directly, and demonstrate the failure on data constructed to
contain it, so a regression fails here rather than showing up as a suspiciously
good result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import load_config
from ml_platform.data import preprocessing as pp

OBSERVATION_END = pd.Timestamp("2014-06-25")
HORIZON = 60


def _cohort(terms: list[int], seed: int = 7) -> pd.DataFrame:
    """Loans whose only real risk driver is a coin flip, not their term.

    Every loan shares one true hazard rate. Any apparent relationship between
    Term and the label in this data is therefore an artifact of the observation
    window, not a property of the loans.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for term in terms:
        for _ in range(300):
            disbursed = pd.Timestamp("2003-01-01")
            # One hazard rate for everyone: a charge-off uniformly spread across
            # the loan's own life.
            defaults = rng.random() < 0.5
            chargeoff = (
                disbursed + pd.to_timedelta(rng.uniform(0, term * 30.44), unit="D")
                if defaults
                else pd.NaT
            )
            rows.append(
                {
                    "MIS_Status": "CHGOFF" if defaults else "P I F",
                    "DisbursementDate": disbursed,
                    "ChgOffDate": chargeoff,
                    "Term": term,
                }
            )
    return pd.DataFrame(rows)


class TestTheArtifactIsRealAndDetectable:
    """Demonstrate the failure mode, so the guard is not protecting a phantom."""

    def test_short_terms_default_far_more_within_the_horizon(self) -> None:
        """With one shared hazard rate, Term still predicts the label.

        Purely because short loans finish inside the window. This is the effect
        the exposure restriction exists to remove.
        """
        cohort = _cohort([24, 36, 240, 300])
        labelled = pp.build_label(cohort, HORIZON, OBSERVATION_END)
        rates = labelled.groupby("Term")["target"].mean()

        assert rates.loc[24] > 0.4
        assert rates.loc[300] < 0.2
        assert rates.loc[24] > 2 * rates.loc[300]

    def test_observed_exposure_varies_without_the_restriction(self) -> None:
        """The mechanism, stated directly.

        A loan is observable for min(its term, the horizon). When that varies
        across loans, the label measures opportunity as well as risk.
        """
        cohort = _cohort([24, 36, 240, 300])
        exposure = cohort["Term"].clip(upper=HORIZON)
        assert exposure.nunique() > 1
        assert exposure.min() == 24

    def test_observed_exposure_is_constant_after_the_restriction(self) -> None:
        """What uniform exposure actually means, and the invariant the fix buys.

        Note what this does *not* claim. The restriction does not flatten the
        relationship between Term and the label, and it should not: within the
        restricted population, term genuinely proxies product type, since
        long-dated SBA lending is collateralised real estate. What it removes is
        the discontinuity, the step at the horizon boundary that comes from loans
        ending before the window does.
        """
        cohort = _cohort([24, 36, 60, 84, 240, 300])
        restricted = pp.restrict_to_uniform_exposure(cohort, HORIZON)
        exposure = restricted["Term"].clip(upper=HORIZON)
        assert exposure.nunique() == 1
        assert exposure.iloc[0] == HORIZON

    def test_the_boundary_cliff_disappears_after_the_restriction(self) -> None:
        """The artifact was a step at the boundary, not a slope.

        Below the boundary the default rate is inflated by loans that finished
        inside the window. The restriction removes exactly those loans.
        """
        cohort = _cohort([48, 84])
        labelled = pp.build_label(cohort, HORIZON, OBSERVATION_END)
        below = labelled.loc[labelled["Term"] == 48, "target"].mean()
        above = labelled.loc[labelled["Term"] == 84, "target"].mean()
        assert below > above

        restricted = pp.restrict_to_uniform_exposure(labelled, HORIZON)
        assert set(restricted["Term"]) == {84}


class TestExposureInvariants:
    def test_no_retained_loan_matures_before_the_horizon_ends(self) -> None:
        """The invariant in one line: every loan is at risk for the full window."""
        frame = pd.DataFrame({"Term": np.arange(0, 600, 1)})
        kept = pp.restrict_to_uniform_exposure(frame, HORIZON)
        assert kept["Term"].min() >= HORIZON

    def test_the_boundary_term_is_retained_not_dropped(self) -> None:
        """A loan of exactly 60 months is exposed for exactly the horizon."""
        frame = pd.DataFrame({"Term": [59, 60, 61]})
        assert pp.restrict_to_uniform_exposure(frame, HORIZON)["Term"].tolist() == [60, 61]

    def test_restriction_removes_only_short_loans(self) -> None:
        frame = pd.DataFrame({"Term": [12, 60, 36, 240, 84]})
        kept = pp.restrict_to_uniform_exposure(frame, HORIZON)
        assert set(kept["Term"]) == {60, 240, 84}

    def test_zero_and_missing_terms_are_removed(self) -> None:
        """810 rows carry Term 0. They cannot be exposed for any horizon."""
        frame = pd.DataFrame({"Term": [0.0, np.nan, 84.0]})
        assert pp.restrict_to_uniform_exposure(frame, HORIZON)["Term"].tolist() == [84.0]

    def test_prepare_applies_the_restriction_when_configured(self) -> None:
        cohort = _cohort([24, 84])
        cohort["ApprovalDate"] = pd.Timestamp("2003-01-01")
        prepared = pp.build_label(cohort, HORIZON, OBSERVATION_END)
        prepared = pp.restrict_to_uniform_exposure(prepared, HORIZON)
        assert (prepared["Term"] >= HORIZON).all()

    def test_prepare_without_a_minimum_term_keeps_short_loans(self) -> None:
        """Passing None must not silently apply a default restriction."""
        cohort = _cohort([24, 84])
        cohort["ApprovalDate"] = pd.Timestamp("2003-01-01")
        labelled = pp.build_label(cohort, HORIZON, OBSERVATION_END)
        assert labelled["Term"].min() == 24


class TestConfigurationCoupling:
    """Horizon and minimum term must move together, or the artifact returns."""

    def test_the_shipped_configuration_couples_them(self) -> None:
        config = load_config("production")
        assert config.min_term_months == config.horizon_months

    @pytest.mark.parametrize("horizon", [24, 36, 60, 84, 120])
    def test_any_horizon_restricts_to_matching_exposure(self, horizon: int) -> None:
        frame = pd.DataFrame({"Term": np.arange(0, 400, 4)})
        kept = pp.restrict_to_uniform_exposure(frame, horizon)
        assert (kept["Term"] >= horizon).all()

    def test_a_mismatched_restriction_leaves_the_artifact(self) -> None:
        """Guards the coupling by showing what a mismatch costs.

        Restricting at 24 months while labelling at 60 leaves loans that mature
        inside the window, and Term regains its spurious relationship.
        """
        cohort = _cohort([24, 36, 240, 300])
        labelled = pp.build_label(cohort, HORIZON, OBSERVATION_END)
        under_restricted = pp.restrict_to_uniform_exposure(labelled, 24)
        rates = under_restricted.groupby("Term")["target"].mean()
        assert rates.max() - rates.min() > 0.2


@pytest.fixture(scope="module")
def real_prepared(real_dataset_available: bool) -> pd.DataFrame:
    """The genuine prepared dataset, or a skip when it has not been downloaded."""
    if not real_dataset_available:
        pytest.skip("real dataset not downloaded; run: python -m ml_platform download")
    from ml_platform.pipelines.train_pipeline import load_prepared_dataset

    frame, _ = load_prepared_dataset(load_config("production"))
    return frame


@pytest.fixture(scope="module")
def real_unrestricted(real_dataset_available: bool) -> pd.DataFrame:
    """The genuine dataset labelled but *without* the exposure restriction.

    This is what the project looked like before the M0 fix, and it is what makes
    the regression test meaningful: the artifact is demonstrated on real data
    rather than only on a constructed cohort.
    """
    if not real_dataset_available:
        pytest.skip("real dataset not downloaded; run: python -m ml_platform download")
    from ml_platform.data import ingestion

    config = load_config("production")
    raw = ingestion.load_raw(config.raw_path)
    return pp.prepare(
        raw,
        config.horizon_months,
        pd.Timestamp(config.observation_end),
        config.target_column,
        min_term_months=None,
    )


@pytest.mark.slow
class TestOnTheRealDataset:
    """The properties that matter, asserted on the genuine register."""

    def test_every_retained_loan_has_uniform_exposure(self, real_prepared: pd.DataFrame) -> None:
        assert real_prepared["Term"].min() >= HORIZON

    def test_the_unrestricted_population_shows_the_boundary_cliff(
        self, real_unrestricted: pd.DataFrame
    ) -> None:
        """The artifact, on genuine data.

        Loans just below the 60-month boundary finish inside the observation
        window, so all of their defaults are captured. Loans just above it do
        not. The result is a step change sitting exactly on the horizon.
        """
        below = real_unrestricted.loc[real_unrestricted["Term"].between(36, 59), "target"].mean()
        above = real_unrestricted.loc[real_unrestricted["Term"].between(61, 84), "target"].mean()
        assert below > 3 * above

    def test_the_restriction_removes_exactly_the_loans_causing_it(
        self, real_unrestricted: pd.DataFrame, real_prepared: pd.DataFrame
    ) -> None:
        """Every dropped row is one that would have matured inside the window."""
        dropped = len(real_unrestricted) - len(real_prepared)
        short = int((real_unrestricted["Term"] < 60).sum())
        assert dropped == short
        assert real_prepared["Term"].min() >= HORIZON

    def test_observed_exposure_is_constant_on_the_real_dataset(
        self, real_prepared: pd.DataFrame
    ) -> None:
        """The invariant the restriction buys, asserted on the genuine register.

        Every retained loan is at risk for exactly the horizon. Term still
        predicts the label afterwards, and legitimately so, because long-dated
        SBA lending is collateralised real estate.
        """
        exposure = real_prepared["Term"].clip(upper=HORIZON)
        assert exposure.nunique() == 1
        assert exposure.iloc[0] == HORIZON

    def test_the_label_horizon_has_elapsed_for_every_row(self, real_prepared: pd.DataFrame) -> None:
        assert (real_prepared["horizon_end"] <= OBSERVATION_END).all()
