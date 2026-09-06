"""Tests for the locked reference file itself.

The reference is the contract a rerun must satisfy. If the file is malformed, the
reproducibility check would pass vacuously, so its shape is asserted here.
"""

from __future__ import annotations

import pytest

from ml_platform.config import load_config
from ml_platform.pipelines.reproduce import (
    PROFILES,
    ReferenceError,
    load_reference,
    tolerance_for,
)

EXPECTED_METRICS = {
    "average_precision",
    "roc_auc",
    "brier_score",
    "brier_skill_score",
    "precision_at_capacity",
    "recall_at_capacity",
    "lift_at_capacity",
    "calibration_ratio",
}
EXPECTED_SPLITS = {"train", "validation", "test"}


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_reference()


class TestReferenceShape:
    def test_both_models_are_locked(self, reference: dict) -> None:
        assert set(reference["models"]) == {"baseline", "candidate"}

    def test_every_model_locks_every_split_and_metric(self, reference: dict) -> None:
        for model in reference["models"].values():
            assert set(model["splits"]) == EXPECTED_SPLITS
            for metrics in model["splits"].values():
                assert set(metrics) == EXPECTED_METRICS

    def test_split_composition_is_locked(self, reference: dict) -> None:
        assert set(reference["splits"]) == EXPECTED_SPLITS | {"production_stream"}
        for composition in reference["splits"].values():
            assert composition["n_rows"] > 0
            assert 0 < composition["n_positive"] < composition["n_rows"]

    def test_dataset_checksum_is_recorded(self, reference: dict) -> None:
        assert len(str(reference["dataset_sha256"])) == 64

    def test_reference_checksum_matches_the_configured_source(self, reference: dict) -> None:
        """The reference must describe the dataset the project actually loads."""
        assert reference["dataset_sha256"] == load_config("production").source["sha256"]


class TestTolerances:
    def test_every_profile_defines_a_default(self, reference: dict) -> None:
        for profile in PROFILES:
            assert "default" in reference["tolerances"][profile]

    def test_strict_is_tighter_than_portable(self, reference: dict) -> None:
        for metric in EXPECTED_METRICS:
            strict = tolerance_for(reference, "strict", metric)
            portable = tolerance_for(reference, "portable", metric)
            assert strict <= portable, metric

    def test_tolerances_are_positive_and_small(self, reference: dict) -> None:
        for profile in PROFILES:
            for metric in EXPECTED_METRICS:
                tolerance = tolerance_for(reference, profile, metric)
                assert 0 < tolerance < 0.05, f"{profile}/{metric}"

    def test_named_metric_overrides_are_honoured(self, reference: dict) -> None:
        table = reference["tolerances"]["portable"]
        if "recall_at_capacity" in table:
            assert (
                tolerance_for(reference, "portable", "recall_at_capacity")
                == (table["recall_at_capacity"])
            )

    def test_unknown_metric_falls_back_to_default(self, reference: dict) -> None:
        default = reference["tolerances"]["strict"]["default"]
        assert tolerance_for(reference, "strict", "not_a_real_metric") == default

    def test_unknown_profile_is_rejected(self, reference: dict) -> None:
        with pytest.raises(ValueError, match="unknown tolerance profile"):
            tolerance_for(reference, "lenient", "roc_auc")


class TestReferenceLoading:
    def test_missing_reference_file_is_an_error(self, tmp_path) -> None:
        with pytest.raises(ReferenceError, match="missing reference file"):
            load_reference(tmp_path)

    def test_malformed_reference_file_is_an_error(self, tmp_path) -> None:
        (tmp_path / "reference.yaml").write_text("other_key: 1", encoding="utf-8")
        with pytest.raises(ReferenceError, match="no top-level"):
            load_reference(tmp_path)
