"""Data validation tests.

Validation runs before training, and a failure must stop the pipeline rather than
produce a quietly wrong model. These tests confirm the schemas actually reject the
things they claim to: missing columns, wrong types, out-of-range values, unhandled
category codes, and labels whose horizon has not yet elapsed.

Each test asserts a *rejection*, not just that clean data passes. A schema that
accepts everything is worse than none, because it looks like protection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml_platform.data import validation as v


def _prepared_row(**overrides: object) -> pd.DataFrame:
    """A minimal frame satisfying PREPARED_SCHEMA."""
    row: dict[str, object] = {
        "ApprovalDate": pd.Timestamp("2004-06-15"),
        "DisbursementDate": pd.Timestamp("2004-07-15"),
        "horizon_end": pd.Timestamp("2009-07-14"),
        "label_available_date": pd.Timestamp("2009-07-14"),
        "target": 0,
        "Term": 84.0,
        "GrAppv": 100_000.0,
        "SBA_Appv": 75_000.0,
        "DisbursementGross": 90_000.0,
        "RevLineCr": "N",
        "LowDoc": "Y",
        "UrbanRural": 1.0,
        "NewExist": 1.0,
        "State": "CA",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _raw_row(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "ApprovalDate": "15-Jun-04",
        "Term": "84",
        "NoEmp": "10",
        "MIS_Status": "P I F",
        "GrAppv": "$100,000.00 ",
        "SBA_Appv": "$75,000.00 ",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestRequiredColumns:
    def test_a_complete_register_passes(self) -> None:
        frame = pd.DataFrame({name: ["x"] for name in v.RAW_COLUMNS})
        v.check_expected_columns(frame)

    def test_a_missing_column_is_reported_by_name(self) -> None:
        """Catches a renamed or restructured upstream mirror."""
        frame = pd.DataFrame({name: ["x"] for name in v.RAW_COLUMNS if name != "MIS_Status"})
        with pytest.raises(v.DataValidationError, match="MIS_Status"):
            v.check_expected_columns(frame)

    def test_several_missing_columns_are_all_reported(self) -> None:
        frame = pd.DataFrame({"ApprovalDate": ["x"]})
        with pytest.raises(v.DataValidationError) as excinfo:
            v.check_expected_columns(frame)
        assert "Term" in str(excinfo.value)
        assert "GrAppv" in str(excinfo.value)

    def test_extra_columns_are_tolerated(self) -> None:
        """Upstream adding a column must not break the pipeline."""
        frame = pd.DataFrame({name: ["x"] for name in v.RAW_COLUMNS})
        frame["SomethingNew"] = "y"
        v.check_expected_columns(frame)


class TestRawSchema:
    def test_a_valid_row_passes(self) -> None:
        assert v.validate(_raw_row(), v.RAW_SCHEMA).passed

    def test_an_unknown_outcome_code_is_rejected(self) -> None:
        """Only the two documented statuses are understood."""
        report = v.validate(_raw_row(MIS_Status="DEFAULTED"), v.RAW_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_null_outcome_is_tolerated(self) -> None:
        """1,997 rows genuinely have no status; they are dropped later, not rejected here."""
        assert v.validate(_raw_row(MIS_Status=None), v.RAW_SCHEMA).passed

    def test_an_out_of_range_term_is_rejected(self) -> None:
        report = v.validate(_raw_row(Term="9000"), v.RAW_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_negative_employee_count_is_rejected(self) -> None:
        report = v.validate(_raw_row(NoEmp="-5"), v.RAW_SCHEMA, raise_on_error=False)
        assert not report.passed


class TestPreparedSchema:
    def test_a_valid_row_passes(self) -> None:
        assert v.validate(_prepared_row(), v.PREPARED_SCHEMA).passed

    def test_a_non_binary_target_is_rejected(self) -> None:
        report = v.validate(_prepared_row(target=2), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_null_target_is_rejected(self) -> None:
        """Every retained row must carry a usable label."""
        report = v.validate(_prepared_row(target=None), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_zero_approval_amount_is_rejected(self) -> None:
        report = v.validate(_prepared_row(GrAppv=0.0), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_negative_disbursement_is_rejected(self) -> None:
        report = v.validate(
            _prepared_row(DisbursementGross=-1.0), v.PREPARED_SCHEMA, raise_on_error=False
        )
        assert not report.passed

    @pytest.mark.parametrize("column", ["RevLineCr", "LowDoc"])
    def test_unnormalised_flag_codes_are_rejected(self, column: str) -> None:
        """The cleaner must fold stray codes to UNK before this point."""
        report = v.validate(_prepared_row(**{column: "T"}), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    @pytest.mark.parametrize("value", ["Y", "N", "UNK"])
    def test_normalised_flag_values_pass(self, value: str) -> None:
        assert v.validate(_prepared_row(RevLineCr=value), v.PREPARED_SCHEMA).passed

    def test_an_unexpected_urban_rural_code_is_rejected(self) -> None:
        report = v.validate(_prepared_row(UrbanRural=7.0), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_zero_new_exist_code_is_rejected(self) -> None:
        """0 means "not stated" and must have become null, not stayed as 0."""
        report = v.validate(_prepared_row(NewExist=0.0), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed

    def test_a_null_new_exist_is_allowed(self) -> None:
        assert v.validate(_prepared_row(NewExist=None), v.PREPARED_SCHEMA).passed

    def test_missing_dates_are_rejected(self) -> None:
        """Both dates drive the label, so neither may be absent."""
        for column in ("ApprovalDate", "DisbursementDate", "horizon_end"):
            report = v.validate(
                _prepared_row(**{column: pd.NaT}), v.PREPARED_SCHEMA, raise_on_error=False
            )
            assert not report.passed, column

    def test_a_string_date_is_coerced_rather_than_rejected(self) -> None:
        """Documents real behaviour: the schema coerces, so type alone proves little.

        This is why the date columns also carry a plausibility range. Without it
        the schema could not detect the century bug, because a mis-parsed date
        column would simply be coerced and pass.
        """
        assert v.validate(_prepared_row(ApprovalDate="2004-06-15"), v.PREPARED_SCHEMA).passed

    def test_a_century_bug_date_is_rejected(self) -> None:
        """The regression this schema exists to catch.

        strptime pivots %y at 1969, so "07-Dec-61" parses as 2061 unless the
        cleaner corrects it. A 2061 approval date must not reach a model.
        """
        report = v.validate(
            _prepared_row(ApprovalDate=pd.Timestamp("2061-12-07")),
            v.PREPARED_SCHEMA,
            raise_on_error=False,
        )
        assert not report.passed

    def test_an_absurdly_early_date_is_rejected(self) -> None:
        report = v.validate(
            _prepared_row(ApprovalDate=pd.Timestamp("1899-01-01")),
            v.PREPARED_SCHEMA,
            raise_on_error=False,
        )
        assert not report.passed

    @pytest.mark.parametrize("column", ["DisbursementDate", "horizon_end", "label_available_date"])
    def test_every_date_column_carries_the_plausibility_range(self, column: str) -> None:
        report = v.validate(
            _prepared_row(**{column: pd.Timestamp("2061-01-01")}),
            v.PREPARED_SCHEMA,
            raise_on_error=False,
        )
        assert not report.passed, column


class TestValidationReporting:
    def test_a_failure_raises_by_default(self) -> None:
        with pytest.raises(v.DataValidationError):
            v.validate(_prepared_row(target=5), v.PREPARED_SCHEMA)

    def test_failures_can_be_collected_instead_of_raised(self) -> None:
        report = v.validate(_prepared_row(target=5), v.PREPARED_SCHEMA, raise_on_error=False)
        assert not report.passed
        assert report.failure_cases

    def test_every_failure_is_collected_not_just_the_first(self) -> None:
        """Lazy validation, so one run surfaces every problem."""
        report = v.validate(
            _prepared_row(target=5, GrAppv=0.0, UrbanRural=9.0),
            v.PREPARED_SCHEMA,
            raise_on_error=False,
        )
        assert len(report.failure_cases) >= 3

    def test_the_summary_states_the_outcome_and_scale(self) -> None:
        report = v.validate(_prepared_row(), v.PREPARED_SCHEMA)
        assert "PASSED" in report.summary()
        assert "1 rows" in report.summary()

    def test_a_failed_summary_says_failed(self) -> None:
        report = v.validate(_prepared_row(target=9), v.PREPARED_SCHEMA, raise_on_error=False)
        assert "FAILED" in report.summary()


class TestLabelHorizonInvariant:
    """The invariant that keeps the delayed-label design honest."""

    def test_elapsed_horizons_pass(self) -> None:
        frame = _prepared_row(horizon_end=pd.Timestamp("2009-07-14"))
        v.assert_label_horizon_elapsed(frame, pd.Timestamp("2014-06-25").date())

    def test_an_unelapsed_horizon_is_rejected(self) -> None:
        """A label that would not yet exist in production must never be used."""
        frame = _prepared_row(horizon_end=pd.Timestamp("2016-01-01"))
        with pytest.raises(v.DataValidationError, match="not yet knowable"):
            v.assert_label_horizon_elapsed(frame, pd.Timestamp("2014-06-25").date())

    def test_the_error_counts_the_offending_rows(self) -> None:
        frame = pd.concat([_prepared_row(horizon_end=pd.Timestamp("2020-01-01"))] * 3)
        with pytest.raises(v.DataValidationError, match="3 rows"):
            v.assert_label_horizon_elapsed(frame, pd.Timestamp("2014-06-25").date())

    def test_the_boundary_itself_is_allowed(self) -> None:
        """A horizon ending exactly on the cutoff is knowable."""
        cutoff = pd.Timestamp("2014-06-25")
        v.assert_label_horizon_elapsed(_prepared_row(horizon_end=cutoff), cutoff.date())
