"""Tests for the two pieces of preprocessing that carry real correctness risk:
century-corrected date parsing, and the fixed-horizon label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_platform.data import preprocessing as pp

OBSERVATION_END = pd.Timestamp("2014-06-25")


class TestTwoDigitDates:
    def test_recent_years_parse_in_the_twentieth_century(self) -> None:
        parsed = pp.parse_two_digit_dates(pd.Series(["28-Feb-97"]), OBSERVATION_END)
        assert parsed.iloc[0] == pd.Timestamp("1997-02-28")

    def test_pre_1969_years_are_pulled_back_a_century(self) -> None:
        """``%y`` pivots at 1969, so 62 would otherwise parse as 2062."""
        parsed = pp.parse_two_digit_dates(pd.Series(["07-Dec-61", "02-Jun-62"]), OBSERVATION_END)
        assert parsed.iloc[0] == pd.Timestamp("1961-12-07")
        assert parsed.iloc[1] == pd.Timestamp("1962-06-02")

    def test_no_parsed_date_exceeds_the_observation_cutoff(self) -> None:
        values = pd.Series(["07-Dec-61", "28-Feb-97", "25-Jun-14", "01-Jan-68"])
        parsed = pp.parse_two_digit_dates(values, OBSERVATION_END)
        assert (parsed <= OBSERVATION_END).all()

    def test_unparseable_values_become_null_rather_than_raising(self) -> None:
        parsed = pp.parse_two_digit_dates(pd.Series(["not-a-date", None]), OBSERVATION_END)
        assert parsed.isna().all()


class TestCurrency:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [('"$60,000.00 "', 60000.0), ("$0.00 ", 0.0), ("$1,234,567.89", 1234567.89)],
    )
    def test_currency_strings_parse_to_floats(self, raw: str, expected: float) -> None:
        result = pp.parse_currency(pd.Series([raw.strip('"')]))
        assert result.iloc[0] == pytest.approx(expected)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Minimal frame carrying only what the label builder reads."""
    return pd.DataFrame(rows)


class TestFixedHorizonLabel:
    def test_chargeoff_inside_the_horizon_is_positive(self) -> None:
        frame = _frame(
            [
                {
                    "MIS_Status": "CHGOFF",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.Timestamp("2007-01-01"),
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert result["target"].iloc[0] == 1

    def test_chargeoff_after_the_horizon_is_negative(self) -> None:
        """A charge-off in month 72 is outside a 60-month horizon."""
        frame = _frame(
            [
                {
                    "MIS_Status": "CHGOFF",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.Timestamp("2011-01-01"),
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert result["target"].iloc[0] == 0

    def test_paid_in_full_is_negative(self) -> None:
        frame = _frame(
            [
                {
                    "MIS_Status": "P I F",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.NaT,
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert result["target"].iloc[0] == 0

    def test_rows_whose_horizon_has_not_elapsed_are_not_observable(self) -> None:
        frame = _frame(
            [
                {
                    "MIS_Status": "P I F",
                    "DisbursementDate": pd.Timestamp("2012-01-01"),
                    "ChgOffDate": pd.NaT,
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert not result["observable"].iloc[0]

    def test_observability_ignores_the_outcome(self) -> None:
        """The inclusion rule must depend only on the calendar.

        This is what keeps the label unbiased. If observability depended on the
        outcome, defaults would be retained more often than healthy loans,
        because they resolve years sooner.
        """
        disbursed = pd.Timestamp("2005-01-01")
        default_row = {
            "MIS_Status": "CHGOFF",
            "DisbursementDate": disbursed,
            "ChgOffDate": pd.Timestamp("2006-01-01"),
            "Term": 84,
        }
        healthy_row = {
            "MIS_Status": "P I F",
            "DisbursementDate": disbursed,
            "ChgOffDate": pd.NaT,
            "Term": 84,
        }
        result = pp.build_label(_frame([default_row, healthy_row]), 60, OBSERVATION_END)
        assert result["observable"].all()
        assert result["horizon_end"].nunique() == 1

    def test_label_available_date_is_the_chargeoff_date_for_defaults(self) -> None:
        chargeoff = pd.Timestamp("2007-03-15")
        frame = _frame(
            [
                {
                    "MIS_Status": "CHGOFF",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": chargeoff,
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert result["label_available_date"].iloc[0] == chargeoff

    def test_label_available_date_is_the_horizon_end_for_negatives(self) -> None:
        frame = _frame(
            [
                {
                    "MIS_Status": "P I F",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.NaT,
                    "Term": 84,
                }
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert result["label_available_date"].iloc[0] == result["horizon_end"].iloc[0]

    def test_label_is_never_available_before_the_prediction(self) -> None:
        """No label may pre-date the disbursement it describes."""
        frame = _frame(
            [
                {
                    "MIS_Status": "CHGOFF",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.Timestamp("2006-06-01"),
                    "Term": 84,
                },
                {
                    "MIS_Status": "P I F",
                    "DisbursementDate": pd.Timestamp("2005-01-01"),
                    "ChgOffDate": pd.NaT,
                    "Term": 84,
                },
            ]
        )
        result = pp.build_label(frame, 60, OBSERVATION_END)
        assert (result["label_available_date"] >= result["DisbursementDate"]).all()


class TestUniformExposure:
    def test_short_term_loans_are_excluded(self) -> None:
        frame = pd.DataFrame({"Term": [12, 36, 59, 60, 84, 240]})
        result = pp.restrict_to_uniform_exposure(frame, 60)
        assert result["Term"].tolist() == [60, 84, 240]

    def test_every_retained_loan_outlives_the_horizon(self) -> None:
        frame = pd.DataFrame({"Term": np.arange(0, 300, 7)})
        result = pp.restrict_to_uniform_exposure(frame, 60)
        assert (result["Term"] >= 60).all()


class TestLeakage:
    def test_outcome_columns_are_dropped(self) -> None:
        frame = pd.DataFrame(
            {
                "MIS_Status": ["CHGOFF"],
                "ChgOffDate": [pd.NaT],
                "ChgOffPrinGr": [1.0],
                "BalanceGross": [0.0],
                "Name": ["ACME"],
                "Term": [84],
                "target": [1],
            }
        )
        result = pp.drop_leakage(frame)
        for column in ("MIS_Status", "ChgOffDate", "ChgOffPrinGr", "BalanceGross", "Name"):
            assert column not in result.columns
        assert "Term" in result.columns
        assert "target" in result.columns
