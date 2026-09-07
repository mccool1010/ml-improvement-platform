"""Request and response schemas.

The request is a **loan application**, in the terms a lender would use, not the
model's feature matrix. Callers should not have to know that the model wants a
guarantee ratio or a NAICS sector; those are derived here by the same
:mod:`ml_platform.features.engineering` code the model was trained with, so the
served features cannot drift from the trained ones.

Fields carry the register's own names because they are the vocabulary of the
dataset, the docs and the run records. Renaming them at the API boundary would
buy nothing and lose traceability.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from ml_platform.data.preprocessing import _normalise_flag

#: Columns the engineered feature set reads. Building a frame with exactly these
#: keeps the API in step with the feature code rather than duplicating its list.
APPLICATION_COLUMNS: tuple[str, ...] = (
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
    "GrAppv",
    "SBA_Appv",
    "DisbursementGross",
    "State",
    "BankState",
    "RevLineCr",
    "LowDoc",
    "UrbanRural",
    "NewExist",
    "NAICS",
    "FranchiseCode",
    "ApprovalDate",
    "DisbursementDate",
)


class LoanApplication(BaseModel):
    """One SBA loan application, as known on the day it is approved.

    Every field is knowable at decision time. Nothing here may describe what
    happened to the loan afterwards.
    """

    model_config = {"extra": "forbid"}

    term_months: Annotated[int, Field(ge=1, le=600, examples=[84])]
    employees: Annotated[int, Field(ge=0, le=10_000, examples=[12])]
    jobs_created: Annotated[int, Field(ge=0, le=10_000, examples=[3])]
    jobs_retained: Annotated[int, Field(ge=0, le=10_000, examples=[8])]

    gross_approved: Annotated[float, Field(gt=0, examples=[250_000.0])]
    sba_approved: Annotated[float, Field(ge=0, examples=[187_500.0])]
    disbursed: Annotated[float, Field(ge=0, examples=[250_000.0])]

    state: Annotated[str, Field(min_length=2, max_length=2, examples=["CA"])]
    bank_state: Annotated[str, Field(min_length=2, max_length=2, examples=["CA"])]

    revolving_line_of_credit: Annotated[str, Field(examples=["N"])] = "UNK"
    low_doc: Annotated[str, Field(examples=["N"])] = "UNK"

    urban_rural: Annotated[Literal[0, 1, 2], Field(examples=[1])] = 0
    new_business: Annotated[Literal[1, 2] | None, Field(examples=[1])] = None

    naics: Annotated[str, Field(examples=["722410"])] = "0"
    franchise_code: Annotated[int, Field(ge=0, examples=[0])] = 0

    approval_date: Annotated[date, Field(examples=["2005-06-15"])]
    disbursement_date: Annotated[date, Field(examples=["2005-07-20"])]

    @field_validator("state", "bank_state")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("naics")
    @classmethod
    def _digits_only(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.isdigit():
            raise ValueError("NAICS must be digits, or '0' when not recorded")
        return cleaned

    @model_validator(mode="after")
    def _check_amounts_and_dates(self) -> LoanApplication:
        if self.sba_approved > self.gross_approved:
            raise ValueError("sba_approved cannot exceed gross_approved")
        if self.disbursement_date < self.approval_date:
            raise ValueError("disbursement_date cannot precede approval_date")
        return self

    def to_frame(self) -> pd.DataFrame:
        """One-row frame in the register's own shape, ready for feature building.

        Flags go through the same normaliser the training data did, so a stray
        code becomes ``UNK`` here exactly as it would have during training.
        """
        row: dict[str, Any] = {
            "Term": float(self.term_months),
            "NoEmp": float(self.employees),
            "CreateJob": float(self.jobs_created),
            "RetainedJob": float(self.jobs_retained),
            "GrAppv": float(self.gross_approved),
            "SBA_Appv": float(self.sba_approved),
            "DisbursementGross": float(self.disbursed),
            "State": self.state,
            "BankState": self.bank_state,
            "RevLineCr": self.revolving_line_of_credit,
            "LowDoc": self.low_doc,
            "UrbanRural": float(self.urban_rural),
            "NewExist": None if self.new_business is None else float(self.new_business),
            "NAICS": self.naics,
            "FranchiseCode": float(self.franchise_code),
            "ApprovalDate": pd.Timestamp(self.approval_date),
            "DisbursementDate": pd.Timestamp(self.disbursement_date),
        }
        frame = pd.DataFrame([row], columns=list(APPLICATION_COLUMNS))
        for column in ("RevLineCr", "LowDoc"):
            frame[column] = _normalise_flag(frame[column])
        return frame


class PredictionRequest(BaseModel):
    """One or more applications to score in a single call."""

    model_config = {"extra": "forbid"}

    applications: Annotated[list[LoanApplication], Field(min_length=1, max_length=1000)]


class ModelInfo(BaseModel):
    """Which model answered, so a prediction can be traced to its origin."""

    name: str
    version: str | None
    alias: str
    feature_set: str
    mlflow_run_id: str | None
    platform_run_id: str | None
    decision_threshold: float
    threshold_source: str


class Prediction(BaseModel):
    """One application's score."""

    default_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    flagged: bool
    threshold: float


class PredictionResponse(BaseModel):
    """Scores plus the identity of the model that produced them."""

    predictions: list[Prediction]
    model: ModelInfo


class HealthResponse(BaseModel):
    """Liveness. The process is running and can answer."""

    status: Literal["ok"]
    service: str


class ReadinessResponse(BaseModel):
    """Readiness. A model is loaded and the service can actually score."""

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    detail: str | None = None
    model: ModelInfo | None = None


class ErrorResponse(BaseModel):
    """A failure the caller can act on."""

    detail: str
