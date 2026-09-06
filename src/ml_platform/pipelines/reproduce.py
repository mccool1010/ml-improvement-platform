"""Reproducibility check.

Trains the baseline and the candidate from the current checkout, then compares
every metric against the locked M1 reference in ``configs/reference.yaml``.

The check is the project's guard against silent drift. A refactor that changes a
number will fail here, and the failure names the metric, the expected value, the
observed value and the tolerance it exceeded. Split composition is compared
exactly, because a difference in row counts is a data or configuration fault
rather than floating-point noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml_platform.config import CONFIG_DIR, load_config
from ml_platform.paths import ensure_dir
from ml_platform.pipelines.train_pipeline import RunRecord, run_training

LOGGER = logging.getLogger(__name__)

REFERENCE_FILE = "reference.yaml"

#: Tolerance profiles. ``strict`` is for a rerun on the reference platform;
#: ``portable`` allows for a different BLAS build on other hardware.
PROFILES = ("strict", "portable")


class ReferenceError(RuntimeError):
    """Raised when the reference file is missing or malformed."""


def load_reference(config_dir: Path | None = None) -> dict[str, Any]:
    """Load the locked reference results."""
    path = (config_dir or CONFIG_DIR) / REFERENCE_FILE
    if not path.exists():
        raise ReferenceError(f"missing reference file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if "reference" not in loaded:
        raise ReferenceError(f"{path} has no top-level 'reference' key")
    return dict(loaded["reference"])


def tolerance_for(reference: dict[str, Any], profile: str, metric: str) -> float:
    """Tolerance for one metric under one profile, falling back to its default."""
    if profile not in PROFILES:
        raise ValueError(f"unknown tolerance profile {profile!r}, expected one of {PROFILES}")
    table: dict[str, Any] = dict(reference["tolerances"][profile])
    if "default" not in table:
        raise ReferenceError(f"tolerance profile {profile!r} has no 'default' entry")
    value = table[metric] if metric in table else table["default"]
    return float(value)


@dataclass
class Deviation:
    """One metric that fell outside tolerance."""

    model: str
    split: str
    metric: str
    expected: float
    observed: float
    tolerance: float

    @property
    def difference(self) -> float:
        return abs(self.observed - self.expected)

    def describe(self) -> str:
        return (
            f"{self.model}/{self.split}/{self.metric}: expected {self.expected:.6f}, "
            f"observed {self.observed:.6f}, difference {self.difference:.2e} "
            f"exceeds tolerance {self.tolerance:.2e}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "split": self.split,
            "metric": self.metric,
            "expected": self.expected,
            "observed": self.observed,
            "difference": self.difference,
            "tolerance": self.tolerance,
        }


@dataclass
class ComparisonResult:
    """Outcome of comparing a set of runs against the reference."""

    profile: str
    passed: bool
    metrics_compared: int
    deviations: list[Deviation] = field(default_factory=list)
    split_mismatches: list[str] = field(default_factory=list)
    exact_matches: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        return (
            f"[{state}] profile={self.profile}: {self.metrics_compared} metrics compared, "
            f"{self.exact_matches} bit-exact, {len(self.deviations)} out of tolerance, "
            f"{len(self.split_mismatches)} split mismatches"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "passed": self.passed,
            "metrics_compared": self.metrics_compared,
            "exact_matches": self.exact_matches,
            "deviations": [d.to_dict() for d in self.deviations],
            "split_mismatches": self.split_mismatches,
            "provenance": self.provenance,
        }


def compare_splits(record: RunRecord, reference: dict[str, Any]) -> list[str]:
    """Compare split composition exactly. Returns a message per mismatch."""
    expected_splits = dict(reference.get("splits") or {})
    observed = {s["name"]: s for s in record.splits}
    mismatches: list[str] = []
    for name, expected in expected_splits.items():
        actual = observed.get(name)
        if actual is None:
            mismatches.append(f"split {name!r} is absent from the run")
            continue
        for key in ("n_rows", "n_positive"):
            if int(actual[key]) != int(expected[key]):
                mismatches.append(
                    f"split {name}.{key}: expected {expected[key]}, observed {actual[key]}"
                )
    return mismatches


def compare_record(
    record: RunRecord, model_key: str, reference: dict[str, Any], profile: str
) -> tuple[list[Deviation], int, int]:
    """Compare one run against the reference. Returns deviations, count, exact count."""
    expected_model = dict(reference["models"][model_key])

    if record.model_name != expected_model["model_name"]:
        raise ReferenceError(
            f"model name changed: reference {expected_model['model_name']!r}, "
            f"run {record.model_name!r}"
        )

    deviations: list[Deviation] = []
    compared = 0
    exact = 0
    for split, expected_metrics in dict(expected_model["splits"]).items():
        observed_metrics = record.metrics.get(split)
        if observed_metrics is None:
            raise ReferenceError(f"run has no metrics for split {split!r}")
        for metric, expected_value in dict(expected_metrics).items():
            observed_value = float(observed_metrics[metric])
            tolerance = tolerance_for(reference, profile, metric)
            difference = abs(observed_value - float(expected_value))
            compared += 1
            if difference == 0.0:
                exact += 1
            elif difference > tolerance:
                deviations.append(
                    Deviation(
                        model=model_key,
                        split=split,
                        metric=metric,
                        expected=float(expected_value),
                        observed=observed_value,
                        tolerance=tolerance,
                    )
                )
    return deviations, compared, exact


def reproduce(
    *,
    environment: str = "production",
    profile: str = "strict",
    save_models: bool = False,
    config_dir: Path | None = None,
) -> ComparisonResult:
    """Train both models and compare them against the locked reference."""
    reference = load_reference(config_dir)
    cfg = load_config(environment)

    records: dict[str, RunRecord] = {}
    for model_key in ("baseline", "candidate"):
        LOGGER.info("training %s", model_key)
        records[model_key] = run_training(
            environment=environment, model_key=model_key, save_model=save_models
        )

    deviations: list[Deviation] = []
    compared = 0
    exact = 0
    for model_key, record in records.items():
        model_deviations, model_compared, model_exact = compare_record(
            record, model_key, reference, profile
        )
        deviations.extend(model_deviations)
        compared += model_compared
        exact += model_exact

    split_mismatches = compare_splits(records["baseline"], reference)

    first = records["baseline"].context
    expected_sha = str(reference.get("dataset_sha256", ""))
    if expected_sha and first.data_sha256 != expected_sha:
        split_mismatches.append(
            f"dataset checksum: expected {expected_sha[:12]}, observed {first.data_sha256[:12]}"
        )

    # Row order is compared exactly. It is the single most informative check here:
    # a mismatch explains a whole page of drifting metrics in one line, because
    # gradient boosting consumes rows in order.
    expected_order = str(reference.get("row_order_sha256", ""))
    observed_order = str(records["baseline"].dataset.get("row_order_sha256", ""))
    if expected_order and observed_order != expected_order:
        split_mismatches.append(
            f"row-order fingerprint: expected {expected_order}, observed {observed_order}; "
            "the data path produced a different row order, so model metrics will differ"
        )

    result = ComparisonResult(
        profile=profile,
        passed=not deviations and not split_mismatches,
        metrics_compared=compared,
        deviations=deviations,
        split_mismatches=split_mismatches,
        exact_matches=exact,
        provenance={
            "git_revision": first.git_revision,
            "git_dirty": first.git_dirty,
            "config_fingerprint": first.config_fingerprint,
            "dataset_sha256": first.data_sha256,
            "lockfile_sha256": first.lockfile_sha256,
            "interpreter": first.interpreter,
            "platform": first.platform,
            "libraries": first.libraries,
            "determinism": first.determinism,
            "reproducible_from_provenance": first.is_reproducible(),
            "row_order_sha256": records["baseline"].dataset.get("row_order_sha256"),
            "reference_milestone": reference.get("milestone"),
            "reference_commit": reference.get("commit"),
        },
    )

    destination = ensure_dir(cfg.benchmark_dir) / f"reproducibility-{first.run_id}.json"
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("wrote reproducibility report to %s", destination)
    return result
