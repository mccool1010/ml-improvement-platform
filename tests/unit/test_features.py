"""Tests for feature construction.

The binding rule is that every feature must be computable at decision time. A
feature that reaches forward, even indirectly, would inflate every metric and
invalidate the whole comparison. These tests check the arithmetic of each derived
feature and assert that no outcome column can reach the model matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.data.preprocessing import IDENTIFIER_COLUMNS, LEAKAGE_COLUMNS
from ml_platform.features import engineering as fe


def _frame(**overrides: object) -> pd.DataFrame:
    """One row carrying every column the feature builder reads."""
    row: dict[str, object] = {
        "Term": 84,
        "NoEmp": 10,
        "CreateJob": 2,
        "RetainedJob": 3,
        "GrAppv": 100_000.0,
        "SBA_Appv": 75_000.0,
        "DisbursementGross": 90_000.0,
        "State": "CA",
        "BankState": "CA",
        "RevLineCr": "N",
        "LowDoc": "Y",
        "UrbanRural": 1.0,
        "NewExist": 1.0,
        "NAICS": "722410",
        "FranchiseCode": 0.0,
        "ApprovalDate": pd.Timestamp("2004-06-15"),
        "DisbursementDate": pd.Timestamp("2004-07-15"),
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestNaicsSector:
    @pytest.mark.parametrize(
        ("code", "sector"),
        [
            ("722410", "accommodation_food"),
            ("451120", "retail"),
            ("621210", "health"),
            ("236220", "construction"),
            ("311111", "manufacturing"),
            ("321113", "manufacturing"),
        ],
    )
    def test_known_codes_map_to_sectors(self, code: str, sector: str) -> None:
        assert fe.naics_sector(pd.Series([code])).iloc[0] == sector

    def test_absent_code_becomes_its_own_level(self) -> None:
        """22.5% of the register has NAICS 0. That is a real gap, not a value to impute."""
        assert fe.naics_sector(pd.Series(["0"])).iloc[0] == "unknown"

    def test_unmapped_prefix_becomes_unknown(self) -> None:
        assert fe.naics_sector(pd.Series(["990000"])).iloc[0] == "unknown"

    def test_short_codes_are_zero_padded_before_slicing(self) -> None:
        """A 4-digit code must not be read as though its first two digits led it."""
        assert fe.naics_sector(pd.Series(["7224"])).iloc[0] == "unknown"


class TestDerivedRatios:
    def test_guarantee_ratio_is_the_sba_share(self) -> None:
        result = fe.add_engineered_features(_frame())
        assert result["sba_guarantee_ratio"].iloc[0] == pytest.approx(0.75)

    def test_guarantee_ratio_is_clipped_to_one(self) -> None:
        result = fe.add_engineered_features(_frame(SBA_Appv=200_000.0))
        assert result["sba_guarantee_ratio"].iloc[0] == 1.0

    def test_zero_approval_yields_null_not_infinity(self) -> None:
        """Dividing by a zero approval must not produce inf and poison the scaler."""
        result = fe.add_engineered_features(_frame(GrAppv=0.0))
        assert pd.isna(result["sba_guarantee_ratio"].iloc[0])
        assert pd.isna(result["disbursement_ratio"].iloc[0])

    def test_disbursement_ratio_compares_disbursed_to_approved(self) -> None:
        result = fe.add_engineered_features(_frame())
        assert result["disbursement_ratio"].iloc[0] == pytest.approx(0.9)

    def test_amount_per_employee_treats_zero_employees_as_one(self) -> None:
        result = fe.add_engineered_features(_frame(NoEmp=0))
        assert result["amount_per_employee"].iloc[0] == pytest.approx(100_000.0)

    def test_jobs_supported_sums_created_and_retained(self) -> None:
        result = fe.add_engineered_features(_frame())
        assert result["jobs_supported"].iloc[0] == 5

    def test_log_approval_is_monotone(self) -> None:
        small = fe.add_engineered_features(_frame(GrAppv=10_000.0))
        large = fe.add_engineered_features(_frame(GrAppv=500_000.0))
        assert small["log_gross_approval"].iloc[0] < large["log_gross_approval"].iloc[0]


class TestCategoricalDerivations:
    def test_same_state_lender_is_detected(self) -> None:
        result = fe.add_engineered_features(_frame(State="CA", BankState="CA"))
        assert result["same_state_lender"].iloc[0] == "same"

    def test_out_of_state_lender_is_detected(self) -> None:
        result = fe.add_engineered_features(_frame(State="CA", BankState="NY"))
        assert result["same_state_lender"].iloc[0] == "different"

    @pytest.mark.parametrize(("code", "expected"), [(0.0, "no"), (1.0, "no"), (44321.0, "yes")])
    def test_franchise_codes_zero_and_one_both_mean_no_franchise(
        self, code: float, expected: str
    ) -> None:
        result = fe.add_engineered_features(_frame(FranchiseCode=code))
        assert result["is_franchise"].iloc[0] == expected

    @pytest.mark.parametrize(
        ("term", "bucket"), [(6, "<=1y"), (24, "1-3y"), (60, "3-5y"), (84, "5-7y"), (300, ">20y")]
    )
    def test_term_buckets_span_the_range(self, term: int, bucket: str) -> None:
        result = fe.add_engineered_features(_frame(Term=term))
        assert result["term_bucket"].iloc[0] == bucket

    def test_approval_month_is_extracted(self) -> None:
        result = fe.add_engineered_features(_frame(ApprovalDate=pd.Timestamp("2004-03-15")))
        assert result["approval_month"].iloc[0] == "3"

    def test_approval_to_disbursement_gap_is_non_negative(self) -> None:
        """A negative gap would be a data error, not a feature."""
        result = fe.add_engineered_features(
            _frame(
                ApprovalDate=pd.Timestamp("2004-07-15"),
                DisbursementDate=pd.Timestamp("2004-06-15"),
            )
        )
        assert result["approval_to_disbursement_days"].iloc[0] == 0


class TestFeatureSets:
    def test_core_is_a_strict_subset_of_engineered(self) -> None:
        core_num, core_cat = fe.feature_columns("core")
        eng_num, eng_cat = fe.feature_columns("engineered")
        assert set(core_num) < set(eng_num)
        assert set(core_cat) < set(eng_cat)

    def test_unknown_feature_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown feature set"):
            fe.feature_columns("everything")

    def test_build_features_returns_exactly_the_declared_columns(self) -> None:
        for feature_set in ("core", "engineered"):
            numeric, categorical = fe.feature_columns(feature_set)
            built = fe.build_features(_frame(), feature_set)
            assert list(built.columns) == [*numeric, *categorical]

    def test_missing_source_column_is_reported_clearly(self) -> None:
        frame = _frame().drop(columns=["GrAppv"])
        with pytest.raises(KeyError, match="needs absent columns"):
            fe.build_features(frame, "core")

    def test_categoricals_never_leave_nulls(self) -> None:
        """A null category would crash the one-hot encoder at serve time."""
        built = fe.build_features(_frame(State=None, NewExist=None), "engineered")
        _, categorical = fe.feature_columns("engineered")
        assert not built[categorical].isna().any().any()

    def test_engineered_adds_columns_without_removing_core_ones(self) -> None:
        core = fe.build_features(_frame(), "core")
        engineered = fe.build_features(_frame(), "engineered")
        assert set(core.columns) < set(engineered.columns)


class TestNoOutcomeLeakage:
    """No feature set may expose a post-decision column."""

    @pytest.mark.parametrize("feature_set", ["core", "engineered"])
    def test_leakage_columns_are_absent_from_feature_sets(self, feature_set: str) -> None:
        numeric, categorical = fe.feature_columns(feature_set)
        declared = set(numeric) | set(categorical)
        assert not declared & set(LEAKAGE_COLUMNS)

    @pytest.mark.parametrize("feature_set", ["core", "engineered"])
    def test_identifier_columns_are_absent_from_feature_sets(self, feature_set: str) -> None:
        numeric, categorical = fe.feature_columns(feature_set)
        declared = set(numeric) | set(categorical)
        assert not declared & set(IDENTIFIER_COLUMNS)

    def test_outcome_columns_present_in_the_source_do_not_reach_the_matrix(self) -> None:
        """Even when the caller passes an unfiltered frame, features stay clean."""
        frame = _frame()
        frame["MIS_Status"] = "CHGOFF"
        frame["ChgOffPrinGr"] = 60_000.0
        frame["ChgOffDate"] = pd.Timestamp("2007-01-01")
        frame["target"] = 1

        built = fe.build_features(frame, "engineered")
        for column in (*LEAKAGE_COLUMNS, "target"):
            assert column not in built.columns

    def test_features_are_finite_for_ordinary_input(self) -> None:
        built = fe.build_features(_frame(), "engineered")
        numeric, _ = fe.feature_columns("engineered")
        assert np.isfinite(built[numeric].to_numpy(dtype=float)).all()
