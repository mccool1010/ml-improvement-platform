"""Building the "current production" window, reproducibly.

Drift detection needs something to compare the training distribution against.
This project has one honest source for that: the 2006 to mid-2009 slice of the
register that M0 quarantined and no model has ever trained on. Slicing it by date
gives a real production window with real covariate shift -- the financial crisis
is inside it -- and slicing is deterministic, so the same dates always produce
the same rows and the same drift report.

That real window is the default. A **controlled scenario** exists alongside it
because a demonstration needs a drift signal that can be dialled up on demand and
asserted against exactly, and because a test that depends on the crisis being
severe enough is a test that fails for the wrong reason. The scenarios here are
explicit, named, seeded transformations of a real window: they say precisely
which features they move and by how much.

Two rules the scenarios follow, both of which matter:

*Only inputs are perturbed.* ``target``, ``MIS_Status``, ``ChgOffDate`` and every
other outcome column are passed through untouched. Manufacturing an outcome to
go with manufactured inputs would be fabricating ground truth, which is the one
thing this milestone must not do.

*The perturbation is a pure function of the seed and the window.* No clock, no
global random state. Running it twice gives byte-identical rows.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml_platform.determinism import ROW_SORT_KIND

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.config import Config

LOGGER = logging.getLogger(__name__)

#: Columns a scenario may never touch. Everything that encodes the outcome, plus
#: the identifiers and dates that decide which rows are observable at all.
PROTECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "target",
        "MIS_Status",
        "ChgOffDate",
        "ChgOffPrinGr",
        "BalanceGross",
        "ApprovalDate",
        "DisbursementDate",
        "LoanNr_ChkDgt",
    }
)


class WindowError(RuntimeError):
    """Raised when a window cannot be produced as asked for."""


@dataclass(frozen=True)
class CurrentWindow:
    """A slice of production data, and how it was produced."""

    frame: pd.DataFrame
    start: date
    end: date
    scenario: str
    seed: int

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def fingerprint(self) -> str:
        """Content and order hashed together, as the project does everywhere.

        Two runs that produce the same window produce the same fingerprint, and
        a drift report carries it so a decision can be tied to exact rows.
        """
        payload = pd.util.hash_pandas_object(self.frame, index=False).to_numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "scenario": self.scenario,
            "seed": self.seed,
            "n_rows": self.n_rows,
            "fingerprint": self.fingerprint(),
        }


def slice_window(frame: pd.DataFrame, start: date, end: date, date_column: str) -> pd.DataFrame:
    """Rows whose approval date falls in ``[start, end]``, in the frame's order."""
    dates = pd.to_datetime(frame[date_column])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[mask].sort_values(date_column, kind=ROW_SORT_KIND).reset_index(drop=True)


# --- controlled scenarios ---------------------------------------------------
#
# Each takes a real window and returns a perturbed copy. The docstrings say what
# moves, because a scenario nobody can read is a scenario nobody can trust.


def _scenario_none(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The real window, untouched. The no-drift control.

    Takes the generator it never uses so every scenario has one signature.
    """
    del rng
    return frame.copy()


def _scenario_mild(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A small shift, of the size that should *not* trip the threshold.

    Loan sizes up 6% with a little noise. This exists to show the detector does
    not fire on any change at all -- a detector that always says drift is as
    useless as one that never does.
    """
    out = frame.copy()
    for column in ("GrAppv", "SBA_Appv", "DisbursementGross"):
        values = pd.to_numeric(out[column], errors="coerce")
        jitter = rng.normal(1.06, 0.01, size=len(out))
        out[column] = values * jitter
    return out


def _scenario_lending_shift(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A credible change in who is borrowing and on what terms.

    Loan amounts up by roughly 70%, guarantee ratios compressed toward the
    statutory ceiling, terms lengthened, and the mix tilted toward new
    businesses in urban areas. This is the shape of a real underwriting-policy
    change: several correlated features moving together, which is exactly the
    population-level signal the detector is supposed to be able to call.
    """
    out = frame.copy()

    gross = pd.to_numeric(out["GrAppv"], errors="coerce") * rng.normal(1.7, 0.08, size=len(out))
    out["GrAppv"] = gross
    # Guarantee ratio pushed up toward the cap, so the derived ratio moves too.
    out["SBA_Appv"] = gross * np.clip(rng.normal(0.86, 0.03, size=len(out)), 0.5, 0.95)
    out["DisbursementGross"] = gross * np.clip(rng.normal(0.97, 0.03, size=len(out)), 0.5, 1.0)

    term = pd.to_numeric(out["Term"], errors="coerce")
    out["Term"] = np.clip(term + rng.integers(24, 96, size=len(out)), 60, 480)

    employees = pd.to_numeric(out["NoEmp"], errors="coerce")
    out["NoEmp"] = np.clip(employees * rng.normal(0.55, 0.05, size=len(out)), 0, None)

    # Categorical mix: more new businesses, more urban.
    to_new = rng.random(len(out)) < 0.45
    out.loc[to_new, "NewExist"] = 2.0
    to_urban = rng.random(len(out)) < 0.5
    out.loc[to_urban, "UrbanRural"] = 1.0
    return out


def _scenario_new_categories(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Values the model has never seen, plus a geographic re-weighting.

    A quarter of rows take a NAICS code from sectors that are rare or absent in
    training, and the state mix is concentrated. This is the failure mode where
    a model keeps producing confident numbers for a population it was never
    shown.
    """
    out = frame.copy()
    unseen = np.array(["928110", "814110", "525910", "999990"])
    picked = rng.random(len(out)) < 0.25
    out.loc[picked, "NAICS"] = rng.choice(unseen, size=int(picked.sum()))

    concentrate = rng.random(len(out)) < 0.4
    out.loc[concentrate, "State"] = "NV"
    return out


#: Named scenarios. `none` is the control; the rest are demonstrations.
SCENARIOS: dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = {
    "none": _scenario_none,
    "mild": _scenario_mild,
    "lending_shift": _scenario_lending_shift,
    "new_categories": _scenario_new_categories,
}


def apply_scenario(frame: pd.DataFrame, scenario: str, seed: int) -> pd.DataFrame:
    """Apply a named scenario deterministically, leaving outcomes alone."""
    if scenario not in SCENARIOS:
        raise WindowError(f"unknown drift scenario {scenario!r}; known: {sorted(SCENARIOS)}")

    before = {column: frame[column].copy() for column in PROTECTED_COLUMNS if column in frame}
    # Seeded per scenario as well as per run, so two scenarios at the same seed
    # do not draw the identical noise. The scenario contributes through a stable
    # digest, never through `hash()`: string hashing is randomised per process,
    # so that would make the window differ between runs of the same command.
    salt = int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4], "big")
    stream = np.random.default_rng([seed, salt])
    out = SCENARIOS[scenario](frame, stream)

    for column, original in before.items():
        if not original.equals(out[column]):
            raise WindowError(
                f"scenario {scenario!r} modified {column!r}, which encodes the outcome or the "
                "observability of a row. Scenarios perturb inputs only."
            )
    return out


def build_current_window(
    prepared: pd.DataFrame,
    config: Config,
    *,
    start: date | None = None,
    end: date | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> CurrentWindow:
    """The production window to compare against, real or perturbed.

    Defaults come from the ``monitoring.window`` block in configuration, so a
    demonstration and a scheduled check run the same code path.
    """
    window_start = start or config.drift_window_start
    window_end = end or config.drift_window_end
    chosen = scenario if scenario is not None else config.drift_scenario
    used_seed = seed if seed is not None else config.seed

    sliced = slice_window(prepared, window_start, window_end, config.split_date_column)
    if sliced.empty:
        raise WindowError(
            f"no rows between {window_start} and {window_end}; the window is outside the data"
        )

    frame = apply_scenario(sliced, chosen, used_seed) if chosen != "none" else sliced.copy()
    window = CurrentWindow(
        frame=frame, start=window_start, end=window_end, scenario=chosen, seed=used_seed
    )
    LOGGER.info(
        "current window %s..%s scenario=%s rows=%d fingerprint=%s",
        window_start,
        window_end,
        chosen,
        window.n_rows,
        window.fingerprint(),
    )
    return window
