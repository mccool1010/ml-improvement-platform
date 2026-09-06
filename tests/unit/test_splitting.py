"""Tests for time-ordered splitting.

The invariant these protect is that no split may contain a row dated after any
row in a later split. A violation would leak the future into training and make
every downstream metric meaningless.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pandas as pd
import pytest

from ml_platform.config import Config, DateWindow
from ml_platform.data.splitting import Split, SplitError, assert_windows_ordered, make_splits


def _config(windows: dict[str, dict[str, str]]) -> Config:
    return Config(
        raw={
            "seed": 42,
            "split": {"strategy": "time_ordered", "date_column": "ApprovalDate", **windows},
        },
        environment="test",
    )


def _frame(dates: list[str], targets: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ApprovalDate": pd.to_datetime(dates),
            "target": targets if targets is not None else [0] * len(dates),
        }
    )


ORDERED = {
    "train": {"start": "2000-01-01", "end": "2003-12-31"},
    "validation": {"start": "2004-01-01", "end": "2004-12-31"},
    "test": {"start": "2005-01-01", "end": "2005-12-31"},
}


class TestDateWindow:
    def test_reversed_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="after end"):
            DateWindow(start=date(2005, 1, 1), end=date(2004, 1, 1))


class TestWindowOrdering:
    def test_ordered_windows_pass(self) -> None:
        assert_windows_ordered(_config(ORDERED))

    def test_overlapping_windows_are_rejected(self) -> None:
        overlapping = {
            "train": {"start": "2000-01-01", "end": "2004-06-30"},
            "validation": {"start": "2004-01-01", "end": "2004-12-31"},
        }
        with pytest.raises(SplitError, match="overlap"):
            assert_windows_ordered(_config(overlapping))

    def test_touching_windows_are_rejected(self) -> None:
        """Sharing a single day still puts the same rows in two splits."""
        touching = {
            "train": {"start": "2000-01-01", "end": "2004-01-01"},
            "validation": {"start": "2004-01-01", "end": "2004-12-31"},
        }
        with pytest.raises(SplitError, match="overlap"):
            assert_windows_ordered(_config(touching))


class TestMakeSplits:
    def test_rows_land_in_the_right_window(self) -> None:
        frame = _frame(["2001-06-01", "2004-06-01", "2005-06-01"])
        splits = make_splits(frame, _config(ORDERED))
        assert splits["train"].n_rows == 1
        assert splits["validation"].n_rows == 1
        assert splits["test"].n_rows == 1

    def test_no_split_overlaps_another_in_time(self) -> None:
        frame = _frame([f"200{y}-0{m}-15" for y in range(0, 6) for m in range(1, 10)])
        splits = make_splits(frame, _config(ORDERED))
        ordered = [splits["train"], splits["validation"], splits["test"]]
        for earlier, later in pairwise(ordered):
            assert earlier.frame["ApprovalDate"].max() < later.frame["ApprovalDate"].min()

    def test_boundary_dates_are_inclusive(self) -> None:
        """Both endpoints of a window belong to it."""
        frame = _frame(["2000-01-01", "2003-12-31", "2004-06-01", "2005-06-01"])
        splits = make_splits(frame, _config(ORDERED))
        assert splits["train"].n_rows == 2

    def test_empty_window_is_an_error(self) -> None:
        frame = _frame(["2001-06-01", "2004-06-01"])
        with pytest.raises(SplitError, match="no rows"):
            make_splits(frame, _config(ORDERED))

    def test_missing_date_column_is_an_error(self) -> None:
        with pytest.raises(SplitError, match="not present"):
            make_splits(pd.DataFrame({"target": [0]}), _config(ORDERED))

    def test_rows_outside_every_window_are_dropped(self) -> None:
        frame = _frame(["1998-01-01", "2001-06-01", "2004-06-01", "2005-06-01", "2009-01-01"])
        splits = make_splits(frame, _config(ORDERED))
        assert sum(s.n_rows for s in splits.values()) == 3


class TestSplitDescribe:
    def test_describe_reports_composition(self) -> None:
        frame = _frame(["2001-01-01", "2001-06-01", "2002-01-01", "2002-06-01"], [1, 0, 0, 0])
        split = Split("train", DateWindow(date(2000, 1, 1), date(2003, 12, 31)), frame)
        described = split.describe()
        assert described["n_rows"] == 4
        assert described["n_positive"] == 1
        assert described["positive_rate"] == pytest.approx(0.25)
