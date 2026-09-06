"""Shared fixtures.

The integration tests need a dataset shaped exactly like the real register but
small enough to train on in seconds. :func:`synthetic_register` builds one in the
raw file's own formats, including the awkward parts: two-digit dates, currency
strings, stray flag codes and a non-numeric fiscal year. Anything the cleaner is
supposed to handle appears here, so the integration path exercises the real
parsing rather than a tidied-up substitute.

Tests that need the genuine 682,428-row dataset are marked ``slow`` and skip
automatically when the raw file has not been downloaded.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import Config
from ml_platform.paths import project_root

OBSERVATION_END = "2014-06-25"
_OBSERVATION_END_DATE = date(2014, 6, 25)
HORIZON_MONTHS = 60
MIN_TERM_MONTHS = 60

#: Enough rows that every split holds both classes and one-hot encoding has
#: categories above its minimum frequency, while still training in seconds.
N_SYNTHETIC_ROWS = 9000

STATES = ["CA", "TX", "NY", "FL", "IL", "OH", "PA", "MI", "GA", "NC"]
NAICS_CODES = ["722410", "451120", "621210", "236220", "541330", "0", "811118", "445110"]


def _fmt_date(value: date | None) -> str:
    """Render a date the way the register does: ``15-Jun-03``."""
    return "" if value is None else value.strftime("%d-%b-%y")


def _fmt_money(value: float) -> str:
    """Render currency the way the register does: ``"$60,000.00 "``."""
    return f"${value:,.2f} "


def build_synthetic_register(n_rows: int = N_SYNTHETIC_ROWS, seed: int = 20260906) -> pd.DataFrame:
    """A raw-shaped register spanning 2000 to mid-2009.

    Default rates rise over time, so the drift the project relies on is present
    in miniature and the time-ordered splits differ from one another.
    """
    rng = np.random.default_rng(seed)
    start = date(2000, 1, 3)
    span_days = (date(2009, 5, 29) - start).days

    approval_offsets = np.sort(rng.integers(0, span_days, size=n_rows))
    rows: list[dict[str, Any]] = []

    for index, offset in enumerate(approval_offsets):
        approval = start + timedelta(days=int(offset))
        disbursement = approval + timedelta(days=int(rng.integers(5, 200)))

        # A mix either side of the horizon, so the exposure restriction has work.
        term = int(rng.choice([12, 36, 48, 60, 84, 120, 180, 240, 300]))

        gross = float(rng.integers(15_000, 900_000))
        guarantee_ratio = float(rng.uniform(0.5, 0.9))
        guarantee = gross * guarantee_ratio
        new_business = int(rng.choice([1, 1, 2, 0]))
        urban_rural = int(rng.choice([0, 1, 1, 2]))

        # Risk rises through the period, mimicking the real crisis effect.
        year_fraction = (approval.year - 2000) / 9.0
        base_risk = 0.03 + 0.30 * year_fraction

        # Risk drivers that survive the uniform-exposure restriction, so a model
        # trained on the restricted population has something real to learn. A
        # fixture with signal only in short terms would leave the integration
        # tests unable to distinguish a working pipeline from a broken one.
        if term < 60:
            base_risk *= 1.4
        base_risk *= 1.0 + 0.9 * (guarantee_ratio - 0.7)
        if new_business == 2:
            base_risk *= 1.5
        if urban_rural == 2:
            base_risk *= 0.6
        if term >= 180:
            base_risk *= 0.5

        defaulted = bool(rng.random() < min(base_risk, 0.75))
        chargeoff: date | None = None
        if defaulted:
            # Some inside the 60-month horizon, some beyond it.
            chargeoff = disbursement + timedelta(days=int(rng.integers(120, 2600)))
            # The register only records events that had happened by the
            # observation cutoff. A charge-off drawn past it simply had not
            # occurred yet, so the loan is recorded as still performing.
            # Without this the fixture emits post-2014 dates, which the
            # century correction then pulls back to 1914.
            if chargeoff > _OBSERVATION_END_DATE:
                defaulted = False
                chargeoff = None

        rows.append(
            {
                "LoanNr_ChkDgt": str(1_000_000_000 + index),
                "Name": f"BUSINESS {index}",
                "City": "SPRINGFIELD",
                "State": str(rng.choice(STATES)),
                "Zip": str(int(rng.integers(10000, 99999))),
                "Bank": "A BANK NA",
                "BankState": str(rng.choice(STATES)),
                "NAICS": str(rng.choice(NAICS_CODES)),
                "ApprovalDate": _fmt_date(approval),
                # One non-numeric fiscal year, as the real file has.
                "ApprovalFY": "1976A" if index == 0 else str(approval.year),
                "Term": str(term),
                "NoEmp": str(int(rng.integers(0, 60))),
                # Includes 0, which means "not stated" and must become null.
                "NewExist": str(new_business),
                "CreateJob": str(int(rng.integers(0, 12))),
                "RetainedJob": str(int(rng.integers(0, 25))),
                "FranchiseCode": str(int(rng.choice([0, 1, 1, 44321]))),
                "UrbanRural": str(urban_rural),
                # Stray codes the normaliser must fold into UNK.
                "RevLineCr": str(rng.choice(["Y", "N", "N", "0", "T", ","])),
                "LowDoc": str(rng.choice(["Y", "N", "N", "C", "1"])),
                "ChgOffDate": _fmt_date(chargeoff),
                "DisbursementDate": _fmt_date(disbursement),
                "DisbursementGross": _fmt_money(gross * float(rng.uniform(0.85, 1.0))),
                "BalanceGross": _fmt_money(0.0),
                "MIS_Status": "CHGOFF" if defaulted else "P I F",
                "ChgOffPrinGr": _fmt_money(gross * 0.6 if defaulted else 0.0),
                "GrAppv": _fmt_money(gross),
                "SBA_Appv": _fmt_money(guarantee),
            }
        )

    return pd.DataFrame(rows)


def _config_payload(raw_dir: Path, filename: str, checksum: str, rows: int) -> dict[str, Any]:
    """A resolved configuration mirroring configs/, pointed at a temporary file."""
    return {
        "project_name": "ml-improvement-platform-test",
        "seed": 42,
        "environment": "test",
        "sample_fraction": 1.0,
        "determinism": {"n_threads": 1, "row_sort_kind": "quicksort"},
        "data": {
            "source": {
                "name": "synthetic-register",
                "url": "file://synthetic",
                "filename": filename,
                "sha256": checksum,
                "expected_rows": rows,
            },
            "raw_dir": str(raw_dir),
            "interim_dir": str(raw_dir / "interim"),
            "processed_dir": str(raw_dir / "processed"),
            "observation_end": OBSERVATION_END,
        },
        "label": {
            "name": "default_within_60m",
            "horizon_months": HORIZON_MONTHS,
            "target_column": "target",
        },
        "population": {"min_term_months": MIN_TERM_MONTHS},
        "split": {
            "strategy": "time_ordered",
            "date_column": "ApprovalDate",
            "train": {"start": "2000-01-01", "end": "2003-12-31"},
            "validation": {"start": "2004-01-01", "end": "2004-12-31"},
            "test": {"start": "2005-01-01", "end": "2005-12-31"},
            "production_stream": {"start": "2006-01-01", "end": "2009-06-30"},
        },
        "artifacts": {
            "model_dir": str(raw_dir / "models"),
            "benchmark_dir": str(raw_dir / "benchmarks"),
        },
        "evaluation": {
            "primary_metric": "average_precision",
            "secondary_metrics": ["roc_auc"],
            "review_capacity": 0.10,
            "report_dir": str(raw_dir / "reports"),
        },
        "baseline": {
            "name": "logistic_regression_baseline",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 500, "C": 1.0, "random_state": 42},
            "feature_set": "core",
        },
        "candidate": {
            "name": "hist_gradient_boosting",
            "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
            "params": {"learning_rate": 0.1, "max_iter": 40, "random_state": 42},
            "feature_set": "engineered",
        },
    }


@pytest.fixture(scope="session")
def synthetic_register() -> pd.DataFrame:
    """A raw-shaped register, built once per session."""
    return build_synthetic_register()


@pytest.fixture(scope="session")
def synthetic_csv(synthetic_register: pd.DataFrame, tmp_path_factory: Any) -> Path:
    """The synthetic register written to disk exactly like the real CSV."""
    directory = tmp_path_factory.mktemp("synthetic_data")
    path = directory / "SyntheticRegister.csv"
    synthetic_register.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_config(synthetic_csv: Path, tmp_path: Path) -> Config:
    """A Config pointing at the synthetic register, writing into a temp directory.

    Paths are absolute, which ``ml_platform.paths.resolve`` passes through
    unchanged, so nothing touches the real project directories.
    """
    raw_dir = tmp_path / "data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / synthetic_csv.name
    destination.write_bytes(synthetic_csv.read_bytes())

    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    rows = sum(1 for _ in destination.open("r", encoding="utf-8")) - 1

    return Config(
        raw=_config_payload(raw_dir, destination.name, checksum, rows),
        environment="test",
    )


@pytest.fixture(scope="session")
def real_dataset_available() -> bool:
    """Whether the genuine register has been downloaded."""
    return (project_root() / "data" / "raw" / "SBAnational.csv").exists()


@pytest.fixture
def requires_real_dataset(real_dataset_available: bool) -> None:
    """Skip a test when the genuine register is absent."""
    if not real_dataset_available:
        pytest.skip("real dataset not downloaded; run: python -m ml_platform download")
