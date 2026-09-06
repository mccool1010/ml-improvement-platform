"""Feature construction.

Two feature sets exist so that "the candidate beat the baseline" is a claim about
something specific.

``core``
    Fields taken almost directly from the register. This is what the baseline
    sees, and it is deliberately unambitious.

``engineered``
    Adds ratios and groupings that encode how SBA lending actually works: the
    share of the loan the government guaranteed, whether the lender is in the
    borrower's state, the industry sector behind the NAICS code, and the gap
    between approval and disbursement.

Every feature here is computable at decision time from the loan application. No
feature may reference an outcome, a date after approval, or a value that is only
known once the loan has run its course.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Fields the baseline model sees.
CORE_NUMERIC: tuple[str, ...] = (
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
    "GrAppv",
    "SBA_Appv",
    "DisbursementGross",
)
CORE_CATEGORICAL: tuple[str, ...] = (
    "State",
    "RevLineCr",
    "LowDoc",
    "UrbanRural",
    "NewExist",
)

#: Additional fields built by :func:`add_engineered_features`.
ENGINEERED_NUMERIC: tuple[str, ...] = (
    "sba_guarantee_ratio",
    "disbursement_ratio",
    "log_gross_approval",
    "amount_per_employee",
    "jobs_supported",
    "approval_to_disbursement_days",
    "term_years",
)
ENGINEERED_CATEGORICAL: tuple[str, ...] = (
    "naics_sector",
    "same_state_lender",
    "is_franchise",
    "term_bucket",
    "approval_month",
)

#: NAICS two-digit prefixes grouped into readable sectors. Several sectors share
#: multiple prefixes, which is why this is a mapping rather than a slice.
_NAICS_SECTORS: dict[str, str] = {
    "11": "agriculture",
    "21": "mining",
    "22": "utilities",
    "23": "construction",
    "31": "manufacturing",
    "32": "manufacturing",
    "33": "manufacturing",
    "42": "wholesale",
    "44": "retail",
    "45": "retail",
    "48": "transport",
    "49": "transport",
    "51": "information",
    "52": "finance",
    "53": "real_estate",
    "54": "professional",
    "55": "management",
    "56": "admin_support",
    "61": "education",
    "62": "health",
    "71": "arts_recreation",
    "72": "accommodation_food",
    "81": "other_services",
    "92": "public_admin",
}

TERM_BINS = [-np.inf, 12, 36, 60, 84, 120, 240, np.inf]
TERM_LABELS = ["<=1y", "1-3y", "3-5y", "5-7y", "7-10y", "10-20y", ">20y"]


def naics_sector(naics: pd.Series) -> pd.Series:
    """Map a NAICS code to a sector name. Missing codes become ``unknown``.

    About 22% of rows carry a NAICS of ``0``, which is a real gap in the register
    rather than something to impute away. It gets its own level.
    """
    prefix = naics.astype("string").str.zfill(6).str[:2]
    return prefix.map(_NAICS_SECTORS).fillna("unknown").astype("string")


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with the engineered columns appended."""
    df = frame.copy()

    gross_approval = df["GrAppv"].replace(0, np.nan)

    # The share of the loan the SBA guaranteed. This is the single most direct
    # expression of how much risk the lender retained.
    df["sba_guarantee_ratio"] = (df["SBA_Appv"] / gross_approval).clip(0, 1)

    # Disbursing materially less than approved signals a changed deal.
    df["disbursement_ratio"] = (df["DisbursementGross"] / gross_approval).clip(0, 5)

    df["log_gross_approval"] = np.log1p(df["GrAppv"].clip(lower=0))
    df["amount_per_employee"] = df["GrAppv"] / df["NoEmp"].clip(lower=1)
    df["jobs_supported"] = df["CreateJob"].fillna(0) + df["RetainedJob"].fillna(0)
    df["term_years"] = df["Term"] / 12.0

    df["approval_to_disbursement_days"] = (
        df["DisbursementDate"] - df["ApprovalDate"]
    ).dt.days.clip(lower=0, upper=3650)

    df["naics_sector"] = naics_sector(df["NAICS"])

    # An out-of-state lender has weaker local knowledge of the borrower.
    df["same_state_lender"] = (
        (df["State"] == df["BankState"]).map({True: "same", False: "different"}).astype("string")
    )

    # FranchiseCode is 0 or 1 for "no franchise", and a real code otherwise.
    df["is_franchise"] = (
        df["FranchiseCode"].fillna(0).gt(1).map({True: "yes", False: "no"}).astype("string")
    )

    df["term_bucket"] = pd.cut(df["Term"], bins=TERM_BINS, labels=TERM_LABELS).astype("string")
    df["approval_month"] = df["ApprovalDate"].dt.month.astype("string")

    return df


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` column names for a named feature set."""
    if feature_set == "core":
        return list(CORE_NUMERIC), list(CORE_CATEGORICAL)
    if feature_set == "engineered":
        return (
            [*CORE_NUMERIC, *ENGINEERED_NUMERIC],
            [*CORE_CATEGORICAL, *ENGINEERED_CATEGORICAL],
        )
    raise ValueError(f"unknown feature set: {feature_set!r}")


def build_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Produce the model input matrix for a named feature set."""
    source = add_engineered_features(frame) if feature_set == "engineered" else frame
    numeric, categorical = feature_columns(feature_set)

    missing = [c for c in (*numeric, *categorical) if c not in source.columns]
    if missing:
        raise KeyError(f"feature set {feature_set!r} needs absent columns: {missing}")

    features = source[[*numeric, *categorical]].copy()
    for column in categorical:
        features[column] = features[column].astype("string").fillna("UNK")
    return features
