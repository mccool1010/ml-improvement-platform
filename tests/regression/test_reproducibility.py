"""Reproducibility regression tests against the locked M1 reference.

The reference in ``configs/reference.yaml`` is the project's contract: a rerun of
the current code on the current data must produce those numbers. These tests are
the automated form of that check.

The full end-to-end reproduction trains both models on the genuine 682,428-row
dataset and is marked ``slow``. The comparison machinery itself is tested on
synthetic records, so a broken comparator is caught in milliseconds rather than
only by the expensive test, and so the failure paths can be exercised at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform.pipelines import reproduce as rp
from ml_platform.pipelines.train_pipeline import RunRecord
from ml_platform.reproducibility import RunContext


def _reference() -> dict[str, Any]:
    """A miniature reference with the same shape as the real one."""
    return {
        "milestone": "TEST",
        "dataset_sha256": "a" * 64,
        "row_order_sha256": "abcd1234abcd1234",
        "tolerances": {"strict": {"default": 1e-6}, "portable": {"default": 1e-3}},
        "splits": {"train": {"n_rows": 100, "n_positive": 10}},
        "models": {
            "baseline": {
                "model_name": "logistic_regression_baseline",
                "splits": {"test": {"average_precision": 0.5, "roc_auc": 0.8}},
            }
        },
    }


def _record(**metric_overrides: float) -> RunRecord:
    metrics = {"average_precision": 0.5, "roc_auc": 0.8}
    metrics.update(metric_overrides)
    context = RunContext(
        run_id="test",
        started_at="2026-09-06T00:00:00+00:00",
        git_revision="abc123",
        git_dirty=False,
        config_fingerprint="ffff",
        environment="test",
        seed=42,
        data_sha256="a" * 64,
        lockfile_sha256="beef",
        determinism={"n_threads": 1},
    )
    return RunRecord(
        context=context,
        model_name="logistic_regression_baseline",
        feature_set="core",
        n_features=10,
        train_seconds=1.0,
        splits=[{"name": "train", "n_rows": 100, "n_positive": 10}],
        metrics={"test": metrics},
        validation_reports=[],
        dataset={"row_order_sha256": "abcd1234abcd1234"},
    )


class TestReferenceLoading:
    def test_the_shipped_reference_loads(self) -> None:
        reference = rp.load_reference()
        assert str(reference["milestone"]).startswith("M1")
        assert reference["dataset_sha256"]

    def test_a_missing_file_is_reported(self, tmp_path: Any) -> None:
        with pytest.raises(rp.ReferenceError, match="missing reference file"):
            rp.load_reference(tmp_path)

    def test_a_file_without_the_top_level_key_is_reported(self, tmp_path: Any) -> None:
        (tmp_path / "reference.yaml").write_text("something_else: 1", encoding="utf-8")
        with pytest.raises(rp.ReferenceError, match="no top-level"):
            rp.load_reference(tmp_path)


class TestTolerances:
    def test_a_named_override_beats_the_default(self) -> None:
        reference = _reference()
        reference["tolerances"]["portable"]["roc_auc"] = 0.05
        assert rp.tolerance_for(reference, "portable", "roc_auc") == 0.05

    def test_an_unnamed_metric_uses_the_default(self) -> None:
        assert rp.tolerance_for(_reference(), "strict", "roc_auc") == 1e-6

    def test_an_unknown_profile_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown tolerance profile"):
            rp.tolerance_for(_reference(), "loose", "roc_auc")

    def test_a_profile_without_a_default_is_reported(self) -> None:
        reference = _reference()
        reference["tolerances"]["strict"] = {"roc_auc": 0.1}
        with pytest.raises(rp.ReferenceError, match="no 'default' entry"):
            rp.tolerance_for(reference, "strict", "average_precision")


class TestMetricComparison:
    def test_an_exact_match_reports_no_deviation(self) -> None:
        deviations, compared, exact = rp.compare_record(
            _record(), "baseline", _reference(), "strict"
        )
        assert deviations == []
        assert compared == 2
        assert exact == 2

    def test_a_drifted_metric_is_caught(self) -> None:
        """The whole point: a refactor that moves a number must fail."""
        deviations, _, _ = rp.compare_record(
            _record(average_precision=0.51), "baseline", _reference(), "strict"
        )
        assert len(deviations) == 1
        assert deviations[0].metric == "average_precision"

    def test_a_difference_inside_tolerance_passes(self) -> None:
        deviations, _, exact = rp.compare_record(
            _record(average_precision=0.5 + 5e-7), "baseline", _reference(), "strict"
        )
        assert deviations == []
        assert exact == 1

    def test_the_portable_profile_is_more_forgiving(self) -> None:
        record = _record(average_precision=0.5005)
        assert rp.compare_record(record, "baseline", _reference(), "strict")[0]
        assert not rp.compare_record(record, "baseline", _reference(), "portable")[0]

    def test_a_renamed_model_is_an_error_not_a_deviation(self) -> None:
        record = _record()
        record.model_name = "something_else"
        with pytest.raises(rp.ReferenceError, match="model name changed"):
            rp.compare_record(record, "baseline", _reference(), "strict")

    def test_a_missing_split_is_an_error(self) -> None:
        record = _record()
        record.metrics = {}
        with pytest.raises(rp.ReferenceError, match="no metrics for split"):
            rp.compare_record(record, "baseline", _reference(), "strict")

    def test_a_deviation_describes_itself_usefully(self) -> None:
        deviations, _, _ = rp.compare_record(
            _record(roc_auc=0.9), "baseline", _reference(), "strict"
        )
        description = deviations[0].describe()
        assert "roc_auc" in description
        assert "expected 0.800000" in description
        assert "observed 0.900000" in description


class TestExactComparisons:
    """Splits, checksums and row order are compared exactly, with no tolerance."""

    def test_matching_splits_produce_no_mismatch(self) -> None:
        assert rp.compare_splits(_record(), _reference()) == []

    def test_a_changed_row_count_is_a_mismatch(self) -> None:
        record = _record()
        record.splits = [{"name": "train", "n_rows": 99, "n_positive": 10}]
        mismatches = rp.compare_splits(record, _reference())
        assert len(mismatches) == 1
        assert "n_rows" in mismatches[0]

    def test_a_changed_positive_count_is_a_mismatch(self) -> None:
        """Same rows, different labels, means the labelling rule moved."""
        record = _record()
        record.splits = [{"name": "train", "n_rows": 100, "n_positive": 11}]
        assert "n_positive" in rp.compare_splits(record, _reference())[0]

    def test_an_absent_split_is_a_mismatch(self) -> None:
        record = _record()
        record.splits = []
        assert "absent" in rp.compare_splits(record, _reference())[0]


class TestComparisonResult:
    def test_a_clean_result_summarises_as_pass(self) -> None:
        result = rp.ComparisonResult(
            profile="strict", passed=True, metrics_compared=48, exact_matches=48
        )
        assert "[PASS]" in result.summary()
        assert "48 bit-exact" in result.summary()

    def test_a_failing_result_summarises_as_fail(self) -> None:
        result = rp.ComparisonResult(profile="strict", passed=False, metrics_compared=48)
        assert "[FAIL]" in result.summary()

    def test_a_result_serialises_for_the_benchmark_report(self) -> None:
        result = rp.ComparisonResult(profile="strict", passed=True, metrics_compared=1)
        payload = result.to_dict()
        assert set(payload) >= {"profile", "passed", "metrics_compared", "deviations"}


@pytest.mark.slow
class TestFullReproduction:
    """The real thing, on the genuine dataset."""

    def test_the_locked_m1_reference_is_reproduced_exactly(
        self, real_dataset_available: bool
    ) -> None:
        if not real_dataset_available:
            pytest.skip("real dataset not downloaded; run: python -m ml_platform download")

        result = rp.reproduce(environment="production", profile="strict", save_models=False)

        assert result.split_mismatches == []
        assert result.deviations == [], [d.describe() for d in result.deviations]
        assert result.passed
        # Bit-exact, not merely within tolerance, on the reference platform.
        assert result.exact_matches == result.metrics_compared
        assert result.metrics_compared == 48
