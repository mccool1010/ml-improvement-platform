"""Cleaning and labelling for the SBA national loan register.

Two things in this module carry most of the project's correctness risk:

1. **Two-digit years.** Dates arrive as ``28-Feb-97``. ``strptime`` pivots ``%y``
   at 1969-2068, so ``07-Jan-62`` parses as 2062 rather than 1962. Those dates
   are pulled back a century.

   The pivot for that correction must be a *fixed* year, not the observation
   cutoff. An earlier version compared against the last approval date,
   2014-06-25, which wrongly rewrote 1,491 charge-off dates from mid-2014 to
   1914. Charge-offs are recorded after the approvals they follow, so they run
   past the last approval legitimately. The resulting negative time-to-default
   then satisfied the in-horizon test and flipped 1,096 labels to positive.

2. **The label.** "Did this loan eventually charge off?" cannot be answered for
   recent loans, and filtering to loans that have *resolved* is biased: defaults
   resolve quickly (median 1,410 days to charge-off) while healthy loans only
   resolve at scheduled maturity. That filter inflates the apparent default rate
   for recent cohorts to implausible levels. Instead this module uses a fixed
   horizon: a loan is positive if it charged off within ``horizon_months`` of
   disbursement, and a loan is only *observable* once that full horizon has
   elapsed. The inclusion rule ignores the outcome, so the label is unbiased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_platform.determinism import ROW_SORT_KIND

DAYS_PER_MONTH = 30.44

#: Any parsed year above this is a two-digit-year artifact and is pulled back a
#: century. ``strptime`` maps ``%y`` values 00-68 to 2000-2068, so the register's
#: earliest years, 1960-1968, arrive as 2060-2068. Nothing in the file is
#: legitimately later than the mid-2010s, so the pivot sits well clear of both:
#: high enough that real 2014 charge-off dates survive, low enough that every
#: mis-centuried value is caught.
CENTURY_PIVOT_YEAR = 2059

DATE_COLUMNS = ("ApprovalDate", "ChgOffDate", "DisbursementDate")
CURRENCY_COLUMNS = ("DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv")

#: Columns knowable only after the lending decision. Dropped before modelling.
#: ``ChgOffPrinGr`` is non-zero for 99.6% of charge-offs and 0.7% of healthy
#: loans, so leaving it in yields a near-perfect and completely useless model.
LEAKAGE_COLUMNS = (
    "MIS_Status",
    "ChgOffDate",
    "ChgOffPrinGr",
    "BalanceGross",
)

#: Free-text identifiers with no predictive value that would leak entity identity.
IDENTIFIER_COLUMNS = ("LoanNr_ChkDgt", "Name", "City", "Zip", "Bank")


def parse_two_digit_dates(series: pd.Series, pivot_year: int = CENTURY_PIVOT_YEAR) -> pd.Series:
    """Parse ``%d-%b-%y`` dates, correcting the century for pre-1969 values.

    ``pivot_year`` must be a fixed year, not the observation cutoff. Charge-offs
    legitimately post-date the last approval in the file, so comparing against
    the cutoff rewrites real 2014 dates to 1914.
    """
    parsed = pd.to_datetime(series, format="%d-%b-%y", errors="coerce")
    overshot = parsed.notna() & (parsed.dt.year > pivot_year)
    return parsed.mask(overshot, parsed - pd.DateOffset(years=100))


def parse_currency(series: pd.Series) -> pd.Series:
    """Turn ``"$60,000.00 "`` into ``60000.0``."""
    cleaned = series.astype("string").str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_flag(series: pd.Series) -> pd.Series:
    """Map a nominally Y/N column onto ``{Y, N, UNK}``.

    ``RevLineCr`` and ``LowDoc`` contain stray values (``0``, ``T``, ``1``, ``,``,
    ``.``, ``A``, ``C``, ``S``...). Anything outside Y/N becomes ``UNK`` rather
    than being silently dropped, so the model can use "this field was garbage"
    as a signal and the validation layer can measure how often it happens.
    """
    upper = series.astype("string").str.strip().str.upper()
    return upper.where(upper.isin(["Y", "N"]), "UNK").fillna("UNK")


def clean_raw(frame: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Type-correct the raw register and normalise its known-messy columns.

    ``observation_end`` is retained for the labelling step; date parsing uses a
    fixed century pivot instead, for the reason in the module docstring.
    """
    del observation_end  # parsing no longer depends on it
    df = frame.copy()

    for column in DATE_COLUMNS:
        df[column] = parse_two_digit_dates(df[column])
    for column in CURRENCY_COLUMNS:
        df[column] = parse_currency(df[column])

    # ApprovalFY holds one non-numeric value ("1976A").
    df["ApprovalFY"] = pd.to_numeric(
        df["ApprovalFY"].astype("string").str.extract(r"(\d{4})")[0], errors="coerce"
    )

    # A handful of rows have no disbursement date; approval is the best stand-in.
    df["DisbursementDate"] = df["DisbursementDate"].fillna(df["ApprovalDate"])

    df["Term"] = pd.to_numeric(df["Term"], errors="coerce")
    for column in ("NoEmp", "CreateJob", "RetainedJob", "NewExist", "UrbanRural", "FranchiseCode"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["RevLineCr"] = _normalise_flag(df["RevLineCr"])
    df["LowDoc"] = _normalise_flag(df["LowDoc"])

    # NewExist is coded 1=existing, 2=new; 0 and NaN both mean "not stated".
    df["NewExist"] = df["NewExist"].where(df["NewExist"].isin([1, 2]))

    df["State"] = df["State"].astype("string").str.strip().str.upper()
    df["BankState"] = df["BankState"].astype("string").str.strip().str.upper()

    return df


def build_label(
    frame: pd.DataFrame,
    horizon_months: int,
    observation_end: pd.Timestamp,
    target_column: str = "target",
) -> pd.DataFrame:
    """Attach the fixed-horizon default label and the delayed-label bookkeeping.

    Adds four columns:

    ``target``
        1 if the loan charged off within the horizon, else 0.
    ``label_available_date``
        When the label would genuinely have been knowable in production. This is
        the charge-off date for in-horizon defaults, and the end of the horizon
        otherwise. Monitoring uses it to decide what may be scored today.
    ``observable``
        Whether the full horizon has elapsed by ``observation_end``. Rows that
        are not observable have no trustworthy label and are excluded.
    ``horizon_end``
        Disbursement plus the horizon.
    """
    df = frame.copy()
    horizon = pd.to_timedelta(horizon_months * DAYS_PER_MONTH, unit="D")

    status = df["MIS_Status"].astype("string").str.strip()
    eventual_default = status.eq("CHGOFF")
    resolved = status.isin(["CHGOFF", "P I F"])

    df["horizon_end"] = df["DisbursementDate"] + horizon
    days_to_chargeoff = (df["ChgOffDate"] - df["DisbursementDate"]).dt.days

    in_horizon_default = eventual_default & days_to_chargeoff.le(horizon_months * DAYS_PER_MONTH)
    df[target_column] = in_horizon_default.fillna(False).astype("int8")

    df["label_available_date"] = pd.to_datetime(
        np.where(in_horizon_default.fillna(False), df["ChgOffDate"], df["horizon_end"])
    )

    # Observability depends only on the calendar, never on the outcome.
    df["observable"] = df["horizon_end"].le(observation_end) & resolved
    return df


def restrict_to_uniform_exposure(frame: pd.DataFrame, min_term_months: int) -> pd.DataFrame:
    """Keep only loans scheduled to stay outstanding for the whole horizon.

    Without this, the target is contaminated by how long the loan was even
    capable of defaulting. A 36-month loan lives its entire life inside a
    60-month window; a 240-month loan is a quarter of the way through. Measured
    on the unrestricted data, the default rate jumps from 41.0% for three-to-five
    year loans to 8.3% for five-to-seven year loans, a cliff sitting exactly on
    the horizon boundary, and ``Term`` alone then reproduces 96% of full-model
    performance. The model would be learning the shape of the observation window.

    Restricting to ``Term >= horizon`` makes every loan at risk for the same 60
    months. It costs 19.4% of rows and leaves ``Term`` as a genuine risk signal:
    long-dated loans in this register are collateralised real-estate lending and
    default far less than short working-capital loans.
    """
    return frame.loc[frame["Term"] >= min_term_months].copy()


def prepare(
    frame: pd.DataFrame,
    horizon_months: int,
    observation_end: pd.Timestamp,
    target_column: str = "target",
    min_term_months: int | None = None,
) -> pd.DataFrame:
    """Clean, label, restrict to uniform exposure, and keep observable rows."""
    cleaned = clean_raw(frame, observation_end)
    labelled = build_label(cleaned, horizon_months, observation_end, target_column)
    observed = labelled.loc[labelled["observable"]].copy()

    if min_term_months is not None:
        observed = restrict_to_uniform_exposure(observed, min_term_months)

    # Row order matters: gradient boosting bins and accumulates in row order, and
    # approval dates tie thousands of times per day. The sort kind is pinned in
    # ml_platform.determinism and the resulting order is fingerprinted into every
    # run record, so a change in ordering fails loudly rather than drifting.
    return observed.sort_values("ApprovalDate", kind=ROW_SORT_KIND).reset_index(drop=True)


def drop_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove post-decision outcome columns and free-text identifiers."""
    to_drop = [c for c in (*LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS) if c in frame.columns]
    return frame.drop(columns=to_drop)
