"""Tests for the MLflow integration.

The property under test throughout is that MLflow is a *recorder*. It copies a
finished run record and never computes anything, it is off unless configured on,
and it can never fail a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_platform import tracking
from ml_platform.config import Config
from ml_platform.pipelines.train_pipeline import RunRecord
from ml_platform.reproducibility import RunContext


def _record(**overrides: Any) -> RunRecord:
    context = RunContext(
        run_id="baseline-20260906T000000Z",
        started_at="2026-09-06T00:00:00+00:00",
        git_revision="abc123",
        git_dirty=False,
        config_fingerprint="ffff0000",
        environment="production",
        seed=42,
        data_sha256="a" * 64,
        lockfile_sha256="beef",
        determinism={"n_threads": 1, "sort_kind": "quicksort"},
        libraries={"sklearn": "1.9.0"},
    )
    payload: dict[str, Any] = {
        "context": context,
        "model_name": "logistic_regression_baseline",
        "feature_set": "core",
        "n_features": 70,
        "train_seconds": 1.25,
        "splits": [{"name": "train", "n_rows": 150158, "n_positive": 4561}],
        "metrics": {
            "test": {
                "average_precision": 0.162915,
                "roc_auc": 0.729854,
                "split": "test",
                "n_rows": 53487,
            }
        },
        "validation_reports": ["raw schema checked"],
        "dataset": {
            "name": "sba-national-7a",
            "label": "default_within_60m",
            "row_order_sha256": "7174370c6f491542",
            "observable_rows": 682421,
        },
    }
    payload.update(overrides)
    return RunRecord(**payload)


def _config(tmp_path: Path, *, enabled: bool = True, log_model: bool = False) -> Config:
    return Config(
        raw={
            "seed": 42,
            "sample_fraction": 1.0,
            "determinism": {"n_threads": 1},
            "label": {"horizon_months": 60, "target_column": "target"},
            "population": {"min_term_months": 60},
            "evaluation": {"review_capacity": 0.10},
            "split": {
                "date_column": "ApprovalDate",
                "train": {"start": "2000-01-01", "end": "2003-12-31"},
                "test": {"start": "2005-01-01", "end": "2005-12-31"},
            },
            "tracking": {
                "enabled": enabled,
                "experiment_name": "test-experiment",
                "backend_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                "artifact_dir": str(tmp_path / "artifacts"),
                "log_model": log_model,
            },
        },
        environment="production",
    )


class TestParams:
    def test_the_model_and_its_settings_are_recorded(self, tmp_path: Path) -> None:
        params = tracking.build_params(_record(), _config(tmp_path))
        assert params["model_name"] == "logistic_regression_baseline"
        assert params["feature_set"] == "core"
        assert params["seed"] == 42
        assert params["n_threads"] == 1

    def test_the_labelling_rules_are_recorded(self, tmp_path: Path) -> None:
        """Two runs with different horizons are different experiments, not noise."""
        params = tracking.build_params(_record(), _config(tmp_path))
        assert params["horizon_months"] == 60
        assert params["min_term_months"] == 60

    def test_split_boundaries_are_recorded(self, tmp_path: Path) -> None:
        params = tracking.build_params(_record(), _config(tmp_path))
        assert params["split_train"] == "2000-01-01..2003-12-31"
        assert params["split_test"] == "2005-01-01..2005-12-31"


class TestMetrics:
    def test_metrics_are_namespaced_by_split(self) -> None:
        metrics = tracking.build_metrics(_record())
        assert metrics["test_average_precision"] == pytest.approx(0.162915)
        assert metrics["test_roc_auc"] == pytest.approx(0.729854)

    def test_non_numeric_values_are_skipped(self) -> None:
        """The record holds strings alongside numbers; MLflow accepts only floats."""
        metrics = tracking.build_metrics(_record())
        assert "test_split" not in metrics

    def test_split_composition_is_recorded(self) -> None:
        metrics = tracking.build_metrics(_record())
        assert metrics["split_train_n_rows"] == 150158.0
        assert metrics["split_train_n_positive"] == 4561.0

    def test_values_are_copied_not_recomputed(self) -> None:
        """The rule that stops MLflow becoming a second source of truth."""
        record = _record()
        record.metrics["test"]["average_precision"] = 0.999
        assert tracking.build_metrics(record)["test_average_precision"] == pytest.approx(0.999)


class TestTags:
    def test_provenance_is_recorded(self) -> None:
        tags = tracking.build_tags(_record(), "baseline")
        assert tags["git_revision"] == "abc123"
        assert tags["config_fingerprint"] == "ffff0000"
        assert tags["dataset_sha256"] == "a" * 64

    def test_the_data_identity_is_recorded(self) -> None:
        tags = tracking.build_tags(_record(), "baseline")
        assert tags["dataset_row_order_sha256"] == "7174370c6f491542"
        assert tags["dataset_name"] == "sba-national-7a"

    def test_the_two_run_ids_are_linkable(self) -> None:
        tags = tracking.build_tags(_record(), "baseline")
        assert tags["platform_run_id"] == "baseline-20260906T000000Z"

    def test_library_versions_are_recorded(self) -> None:
        assert tracking.build_tags(_record(), "baseline")["lib_sklearn"] == "1.9.0"

    def test_the_model_key_distinguishes_baseline_from_candidate(self) -> None:
        assert tracking.build_tags(_record(), "candidate")["model_key"] == "candidate"


class TestDisabledByDefault:
    def test_a_config_without_a_tracking_block_logs_nothing(self) -> None:
        """This is what keeps every pre-M4 test free of MLflow side effects."""
        bare = Config(raw={}, environment="test")
        assert not bare.tracking_enabled
        assert tracking.log_run_record(_record(), None, bare) is None

    def test_explicitly_disabled_tracking_logs_nothing(self, tmp_path: Path) -> None:
        config = _config(tmp_path, enabled=False)
        assert tracking.log_run_record(_record(), None, config) is None
        assert not (tmp_path / "mlflow.db").exists()


class TestFailureIsolation:
    def test_a_tracking_failure_returns_none_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training is the deliverable; a recorder must not be able to fail it."""
        config = _config(tmp_path)

        def _explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("tracking backend unreachable")

        monkeypatch.setattr(tracking, "_ensure_experiment", _explode)
        record = _record()
        assert tracking.log_run_record(record, None, config) is None
        assert record.mlflow_run_id is None


class TestLoggingARun:
    """One real MLflow run against a temporary SQLite store."""

    @pytest.fixture
    def logged(self, tmp_path: Path) -> tuple[str, Any]:
        import mlflow

        config = _config(tmp_path)
        record = _record()
        run_id = tracking.log_run_record(record, None, config, model_key="baseline")
        assert run_id is not None, "expected a run id"
        assert record.mlflow_run_id == run_id

        mlflow.set_tracking_uri(config.tracking_uri)
        return run_id, mlflow.MlflowClient().get_run(run_id)

    def test_the_run_is_marked_finished(self, logged: tuple[str, Any]) -> None:
        assert logged[1].info.status == "FINISHED"

    def test_parameters_reach_mlflow(self, logged: tuple[str, Any]) -> None:
        params = logged[1].data.params
        assert params["model_name"] == "logistic_regression_baseline"
        assert params["seed"] == "42"

    def test_metrics_reach_mlflow(self, logged: tuple[str, Any]) -> None:
        metrics = logged[1].data.metrics
        assert metrics["test_average_precision"] == pytest.approx(0.162915)
        assert metrics["test_roc_auc"] == pytest.approx(0.729854)

    def test_tags_reach_mlflow(self, logged: tuple[str, Any]) -> None:
        assert logged[1].data.tags["model_key"] == "baseline"

    def test_the_run_record_is_attached_as_an_artifact(self, logged: tuple[str, Any]) -> None:
        import mlflow

        paths = {a.path for a in mlflow.MlflowClient().list_artifacts(logged[0])}
        assert "run_record.json" in paths

    def test_hyperparameters_are_prefixed(self, tmp_path: Path) -> None:
        import mlflow

        config = _config(tmp_path)
        run_id = tracking.log_run_record(
            _record(), None, config, hyperparameters={"max_iter": 2000, "C": 1.0}
        )
        assert run_id is not None
        mlflow.set_tracking_uri(config.tracking_uri)
        params = mlflow.MlflowClient().get_run(run_id).data.params
        assert params["hp_max_iter"] == "2000"
        assert params["hp_C"] == "1.0"

    def test_separate_calls_produce_separate_runs(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        first = tracking.log_run_record(_record(), None, config, model_key="baseline")
        second = tracking.log_run_record(_record(), None, config, model_key="candidate")
        assert first is not None and second is not None
        assert first != second
