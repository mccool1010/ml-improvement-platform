"""Time-ordered splitting.

Random splitting would leak the future into the past: loans approved in 2008
would help train a model evaluated on 2004. Every split here is a contiguous,
ordered, non-overlapping window of approval dates, which is how the model would
actually be built and used.

The ``production_stream`` window is never touched during model development. It is
replayed later as live traffic for drift detection and retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import pandas as pd

from ml_platform.config import Config, DateWindow


@dataclass(frozen=True)
class Split:
    """One named, time-bounded slice of the dataset."""

    name: str
    window: DateWindow
    frame: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def positive_rate(self, target: str = "target") -> float:
        return float(self.frame[target].mean()) if len(self.frame) else float("nan")

    def describe(self, target: str = "target") -> dict[str, Any]:
        return {
            "name": self.name,
            "start": str(self.window.start),
            "end": str(self.window.end),
            "n_rows": self.n_rows,
            "n_positive": int(self.frame[target].sum()) if len(self.frame) else 0,
            "positive_rate": round(self.positive_rate(target), 6),
        }


class SplitError(RuntimeError):
    """Raised when the configured split windows are unusable."""


def _slice(frame: pd.DataFrame, date_column: str, window: DateWindow) -> pd.DataFrame:
    dates = frame[date_column]
    mask = (dates >= pd.Timestamp(window.start)) & (dates <= pd.Timestamp(window.end))
    return frame.loc[mask].copy()


def assert_windows_ordered(config: Config) -> None:
    """Windows must not overlap, or the test set contains training rows."""
    ordered = sorted(
        ((name, config.window(name)) for name in config.split_names),
        key=lambda item: item[1].start,
    )
    for (name_a, win_a), (name_b, win_b) in pairwise(ordered):
        if win_b.start <= win_a.end:
            raise SplitError(
                f"split windows {name_a} ({win_a.start}..{win_a.end}) and "
                f"{name_b} ({win_b.start}..{win_b.end}) overlap"
            )


def make_splits(frame: pd.DataFrame, config: Config) -> dict[str, Split]:
    """Cut ``frame`` into the configured time-ordered windows."""
    date_column = config.split_date_column
    if date_column not in frame.columns:
        raise SplitError(f"split date column {date_column!r} is not present")

    assert_windows_ordered(config)

    splits: dict[str, Split] = {}
    for name in config.split_names:
        window = config.window(name)
        splits[name] = Split(name, window, _slice(frame, date_column, window))

    empty = [name for name, split in splits.items() if split.n_rows == 0]
    if empty:
        raise SplitError(f"split windows produced no rows: {empty}")
    return splits


def subsample(split: Split, fraction: float, seed: int) -> Split:
    """Take a subsample, used only for fast local iteration.

    Order is restored afterwards so downstream code can still rely on the frame
    being sorted by approval date.
    """
    if fraction >= 1.0 or split.n_rows == 0:
        return split
    sampled = split.frame.sample(frac=fraction, random_state=seed).sort_values("ApprovalDate")
    return Split(split.name, split.window, sampled)
