"""Integration tests for the complete training and evaluation path.

These run the real pipeline end to end, from a raw CSV in the register's own
formats through cleaning, labelling, validation, splitting, fitting, evaluation
and artifact writing. Only the dataset is substituted, for speed; every stage of
the code is the production one.

What they protect is the wiring. Unit tests confirm each stage is correct in
isolation; these confirm the stages are connected in the right order, that
validation genuinely blocks training, and that a run leaves behind an artifact
carrying enough provenance to be treated as evidence.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

from ml_platform.config import Config
from ml_platform.data.splitting import make_splits
from ml_platform.data.validation import DataValidationError
from ml_platform.pipelines.train_pipeline import (
    DEVELOPMENT_SPLITS,
    RunRecord,
    load_prepared_dataset,
    run_training,
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
    "n_rows",
    "n_positive",
    "positive_rate",
}


@pytest.fixture(scope="module")
def _module_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("integration")


class TestDataPathIntegration:
    def test_the_raw_register_survives_the_whole_data_path(self, synthetic_config: Config) -> None:
        prepared, checksum = load_prepared_dataset(synthetic_config)
        assert len(prepared) > 0
        assert checksum == synthetic_config.source["sha256"]

    def test_preparation_produces_the_label_bookkeeping_columns(
        self, synthetic_config: Config
    ) -> None:
        prepared, _ = load_prepared_dataset(synthetic_config)
        for column in ("target", "horizon_end", "label_available_date", "observable"):
            assert column in prepared.columns

    def test_every_prepared_row_satisfies_both_project_invariants(
        self, synthetic_config: Config
    ) -> None:
        """Elapsed horizon and uniform exposure, checked on real pipeline output."""
        prepared, _ = load_prepared_dataset(synthetic_config)
        assert (prepared["horizon_end"] <= pd.Timestamp(synthetic_config.observation_end)).all()
        assert (prepared["Term"] >= synthetic_config.min_term_months).all()

    def test_rows_are_ordered_by_approval_date(self, synthetic_config: Config) -> None:
        prepared, _ = load_prepared_dataset(synthetic_config)
        assert prepared["ApprovalDate"].is_monotonic_increasing

    def test_splits_are_disjoint_and_time_ordered(self, synthetic_config: Config) -> None:
        prepared, _ = load_prepared_dataset(synthetic_config)
        splits = make_splits(prepared, synthetic_config)
        ordered = [splits[n] for n in ("train", "validation", "test", "production_stream")]
        for earlier, later in pairwise(ordered):
            assert earlier.frame["ApprovalDate"].max() < later.frame["ApprovalDate"].min()

    def test_every_split_contains_both_classes(self, synthetic_config: Config) -> None:
        """Otherwise evaluation is undefined and the run is not comparable."""
        prepared, _ = load_prepared_dataset(synthetic_config)
        for split in make_splits(prepared, synthetic_config).values():
            assert split.frame["target"].nunique() == 2, split.name

    def test_a_row_count_mismatch_stops_the_run(self, synthetic_config: Config) -> None:
        """Guards against a truncated or extended upstream file."""
        altered = Config(
            raw={
                **synthetic_config.raw,
                "data": {
                    **synthetic_config.raw["data"],
                    "source": {**synthetic_config.source, "expected_rows": 12},
                },
            },
            environment="test",
        )
        with pytest.raises(DataValidationError, match="expected 12 raw rows"):
            load_prepared_dataset(altered)

    def test_a_checksum_mismatch_stops_the_run(self, synthetic_config: Config) -> None:
        """A changed upstream mirror must fail, not silently alter results."""
        from ml_platform.data.ingestion import DataIntegrityError

        altered = Config(
            raw={
                **synthetic_config.raw,
                "data": {
                    **synthetic_config.raw["data"],
                    "source": {**synthetic_config.source, "sha256": "0" * 64},
                },
            },
            environment="test",
        )
        with pytest.raises(DataIntegrityError, match="checksum mismatch"):
            load_prepared_dataset(altered)


class TestTrainingRun:
    @pytest.fixture
    def record(self, synthetic_config: Config) -> RunRecord:
        """One baseline run per assertion, against a fresh temporary directory."""
        return run_training(config=synthetic_config, model_key="baseline", save_model=True)

    def test_the_run_completes_and_names_its_model(self, record: RunRecord) -> None:
        assert record.model_name == "logistic_regression_baseline"
        assert record.feature_set == "core"

    def test_every_development_split_is_evaluated(self, record: RunRecord) -> None:
        assert set(record.metrics) == set(DEVELOPMENT_SPLITS)

    def test_the_production_stream_is_not_evaluated(self, record: RunRecord) -> None:
        """It is quarantined for the drift and retraining milestones."""
        assert "production_stream" not in record.metrics

    def test_every_split_reports_the_full_metric_set(self, record: RunRecord) -> None:
        for split, metrics in record.metrics.items():
            assert set(metrics) >= EXPECTED_METRICS, split

    def test_metrics_are_in_their_valid_ranges(self, record: RunRecord) -> None:
        for split, metrics in record.metrics.items():
            assert 0.0 <= metrics["average_precision"] <= 1.0, split
            assert 0.0 <= metrics["roc_auc"] <= 1.0, split
            assert 0.0 <= metrics["brier_score"] <= 1.0, split
            assert 0.0 <= metrics["recall_at_capacity"] <= 1.0, split

    def test_the_model_beats_random_ranking(self, record: RunRecord) -> None:
        """A pipeline that wires up correctly but learns nothing would pass everything else."""
        assert record.metrics["test"]["roc_auc"] > 0.55

    def test_split_composition_is_recorded_for_every_window(self, record: RunRecord) -> None:
        recorded = {s["name"] for s in record.splits}
        assert recorded == {"train", "validation", "test", "production_stream"}

    def test_the_dataset_block_records_the_labelling_rules(self, record: RunRecord) -> None:
        assert record.dataset["horizon_months"] == 60
        assert record.dataset["min_term_months"] == 60
        assert len(record.dataset["row_order_sha256"]) == 16

    def test_provenance_is_complete_enough_to_be_evidence(self, record: RunRecord) -> None:
        context = record.context
        assert context.config_fingerprint
        assert context.data_sha256
        assert context.determinism["n_threads"] >= 1
        assert context.libraries["sklearn"]
        assert context.interpreter


class TestArtifacts:
    def test_a_run_writes_a_report_and_a_model(self, synthetic_config: Config) -> None:
        record = run_training(config=synthetic_config, model_key="baseline", save_model=True)

        reports = list(synthetic_config.report_dir.glob("*.json"))
        models = list(synthetic_config.model_dir.glob("*.joblib"))
        assert len(reports) == 1
        assert len(models) == 1
        assert record.model_path is not None

    def test_the_report_is_valid_json_carrying_the_metrics(self, synthetic_config: Config) -> None:
        run_training(config=synthetic_config, model_key="baseline", save_model=False)
        report = next(iter(synthetic_config.report_dir.glob("*.json")))
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["metrics"]["test"]["average_precision"] >= 0
        assert payload["context"]["seed"] == synthetic_config.seed

    def test_the_shipped_artifact_directories_sit_inside_the_project(self) -> None:
        """Why recorded paths are relative in a real run.

        `relative_to_root` falls back to an absolute path for anything outside
        the project, which is correct but means the guarantee only holds while
        the configured directories are inside it. That is what this asserts.
        """
        from ml_platform.config import load_config
        from ml_platform.paths import project_root

        config = load_config("production")
        for directory in (config.model_dir, config.report_dir, config.benchmark_dir):
            assert directory.is_relative_to(project_root())

    def test_a_model_written_inside_the_project_is_recorded_relatively(
        self, synthetic_config: Config
    ) -> None:
        """An absolute path would make records incomparable across machines."""
        from ml_platform.paths import project_root, relative_to_root

        rendered = relative_to_root(project_root() / "models" / "example.joblib")
        assert rendered == "models/example.joblib"

        record = run_training(config=synthetic_config, model_key="baseline", save_model=True)
        assert record.model_path is not None
        assert record.model_path.endswith(".joblib")

    def test_no_model_file_appears_when_saving_is_disabled(self, synthetic_config: Config) -> None:
        record = run_training(config=synthetic_config, model_key="baseline", save_model=False)
        assert record.model_path is None
        assert not list(synthetic_config.model_dir.glob("*.joblib"))

    def test_a_saved_model_reloads_and_scores(self, synthetic_config: Config) -> None:
        """The artifact must be usable, not merely present."""
        from ml_platform.models.train import load_model
        from ml_platform.paths import project_root

        record = run_training(config=synthetic_config, model_key="baseline", save_model=True)
        assert record.model_path is not None
        loaded = load_model(project_root() / record.model_path)

        prepared, _ = load_prepared_dataset(synthetic_config)
        from ml_platform.features.engineering import build_features

        scores = loaded.pipeline.predict_proba(build_features(prepared.head(50), "core"))[:, 1]
        assert len(scores) == 50
        assert ((scores >= 0) & (scores <= 1)).all()


class TestBothModelsTrain:
    @pytest.mark.parametrize("model_key", ["baseline", "candidate"])
    def test_each_configured_model_runs_end_to_end(
        self, synthetic_config: Config, model_key: str
    ) -> None:
        record = run_training(config=synthetic_config, model_key=model_key, save_model=False)
        assert record.metrics["test"]["n_rows"] > 0

    def test_the_candidate_uses_more_features_than_the_baseline(
        self, synthetic_config: Config
    ) -> None:
        baseline = run_training(config=synthetic_config, model_key="baseline", save_model=False)
        candidate = run_training(config=synthetic_config, model_key="candidate", save_model=False)
        assert candidate.n_features > baseline.n_features


class TestValidationBlocksTraining:
    def test_a_broken_register_fails_before_a_model_is_fitted(
        self, synthetic_config: Config, tmp_path: Path
    ) -> None:
        """Validation must stop the pipeline, not merely warn."""
        import hashlib

        corrupt = pd.read_csv(synthetic_config.raw_path, dtype=str)
        corrupt = corrupt.drop(columns=["MIS_Status"])
        destination = tmp_path / "corrupt" / "register.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        corrupt.to_csv(destination, index=False)

        altered = Config(
            raw={
                **synthetic_config.raw,
                "data": {
                    **synthetic_config.raw["data"],
                    "raw_dir": str(destination.parent),
                    "source": {
                        **synthetic_config.source,
                        "filename": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "expected_rows": len(corrupt),
                    },
                },
            },
            environment="test",
        )
        with pytest.raises(DataValidationError, match="MIS_Status"):
            run_training(config=altered, model_key="baseline", save_model=False)
