"""Schema validation for the SBA register.

Validation runs *before* training, and a failure stops the pipeline rather than
producing a quietly wrong model. Two schemas are enforced:

``RAW_SCHEMA``
    What must be true of the file as downloaded. Catches a changed upstream
    mirror, a truncated download, or renamed columns.

``PREPARED_SCHEMA``
    What must be true after cleaning and labelling. Catches parsing regressions,
    such as the two-digit-year century bug or an inverted observability rule.

The messy real-world columns are asserted against the value sets the cleaner is
supposed to produce, so a change in upstream coding surfaces here rather than as
an unexplained metric drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

RAW_COLUMNS: tuple[str, ...] = (
    "LoanNr_ChkDgt",
    "Name",
    "City",
    "State",
    "Zip",
    "Bank",
    "BankState",
    "NAICS",
    "ApprovalDate",
    "ApprovalFY",
    "Term",
    "NoEmp",
    "NewExist",
    "CreateJob",
    "RetainedJob",
    "FranchiseCode",
    "UrbanRural",
    "RevLineCr",
    "LowDoc",
    "ChgOffDate",
    "DisbursementDate",
    "DisbursementGross",
    "BalanceGross",
    "MIS_Status",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
)

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "ApprovalDate": pa.Column(str, nullable=False),
        "Term": pa.Column(pa.Int64, pa.Check.in_range(0, 600), nullable=False, coerce=True),
        "NoEmp": pa.Column(pa.Int64, pa.Check.ge(0), nullable=False, coerce=True),
        "MIS_Status": pa.Column(str, pa.Check.isin(["P I F", "CHGOFF"]), nullable=True),
        "GrAppv": pa.Column(str, nullable=False),
        "SBA_Appv": pa.Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
    name="sba_raw_register",
)

PREPARED_SCHEMA = pa.DataFrameSchema(
    {
        "ApprovalDate": pa.Column("datetime64[ns]", nullable=False),
        "DisbursementDate": pa.Column("datetime64[ns]", nullable=False),
        "horizon_end": pa.Column("datetime64[ns]", nullable=False),
        "label_available_date": pa.Column("datetime64[ns]", nullable=False),
        "target": pa.Column(pa.Int8, pa.Check.isin([0, 1]), nullable=False, coerce=True),
        "Term": pa.Column(pa.Float64, pa.Check.in_range(0, 600), nullable=True, coerce=True),
        "GrAppv": pa.Column(pa.Float64, pa.Check.gt(0), nullable=False, coerce=True),
        "SBA_Appv": pa.Column(pa.Float64, pa.Check.ge(0), nullable=False, coerce=True),
        "DisbursementGross": pa.Column(pa.Float64, pa.Check.ge(0), nullable=False, coerce=True),
        "RevLineCr": pa.Column(str, pa.Check.isin(["Y", "N", "UNK"]), coerce=True),
        "LowDoc": pa.Column(str, pa.Check.isin(["Y", "N", "UNK"]), coerce=True),
        "UrbanRural": pa.Column(pa.Float64, pa.Check.isin([0, 1, 2]), nullable=True, coerce=True),
        "NewExist": pa.Column(pa.Float64, pa.Check.isin([1, 2]), nullable=True, coerce=True),
        "State": pa.Column(str, nullable=True, coerce=True),
    },
    strict=False,
    coerce=True,
    name="sba_prepared",
)


@dataclass
class ValidationReport:
    """Outcome of a validation run, recorded alongside training results."""

    schema: str
    passed: bool
    rows: int
    failure_cases: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        state = "PASSED" if self.passed else "FAILED"
        return f"[{state}] {self.schema}: {self.rows} rows, {len(self.failure_cases)} failure cases"


class DataValidationError(RuntimeError):
    """Raised when a dataset does not satisfy its schema."""


def validate(
    frame: pd.DataFrame, schema: pa.DataFrameSchema, *, raise_on_error: bool = True
) -> ValidationReport:
    """Validate ``frame``, collecting every failure rather than stopping at the first."""
    try:
        schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        cases = exc.failure_cases.head(50).to_dict(orient="records")
        report = ValidationReport(str(schema.name), False, len(frame), cases)
        if raise_on_error:
            raise DataValidationError(f"{report.summary()}\n{exc.failure_cases.head(20)}") from exc
        return report
    return ValidationReport(str(schema.name), True, len(frame))


def check_expected_columns(frame: pd.DataFrame) -> None:
    """Fail loudly if the upstream mirror changed shape."""
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"raw file is missing expected columns: {missing}")


def assert_label_horizon_elapsed(frame: pd.DataFrame, observation_end: date) -> None:
    """Every retained row must have a fully elapsed label horizon.

    This is the invariant that keeps the delayed-label design honest. If it ever
    fails, the dataset contains labels that would not have existed in production.
    """
    cutoff = pd.Timestamp(observation_end)
    late = frame.loc[frame["horizon_end"] > cutoff]
    if not late.empty:
        raise DataValidationError(
            f"{len(late)} rows have a label horizon ending after the observation "
            f"cutoff {cutoff.date()}; their labels are not yet knowable"
        )
