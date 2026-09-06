"""Tests for production resolution, registration and the promotion pipeline.

The property these protect is that the gate report, not MLflow and not recency,
decides what becomes production.
"""

from __future__ import annotations

from typing import Any

import pytest

from ml_platform.config import Config
from ml_platform.pipelines.train_pipeline import RunRecord
from ml_platform.promotion import compare, registry
from ml_platform.promotion.gates import GateReport, GateResult
from ml_platform.reproducibility import RunContext

VALIDATION_METRICS: dict[str, Any] = {
    "average_precision": 0.732031,
    "roc_auc": 0.973004,
    "brier_score": 0.022446,
    "brier_skill_score": 0.501043,
    "recall_at_capacity": 0.868513,
    "inference_seconds": 0.83,
    "n_rows": 54284,
}


def _record(mlflow_run_id: str | None = "mlrun123") -> RunRecord:
    context = RunContext(
        run_id="candidate-20260907T000000Z",
        started_at="2026-09-07T00:00:00+00:00",
        git_revision="abc123",
        git_dirty=False,
        config_fingerprint="ffff0000",
        environment="production",
        seed=42,
        data_sha256="a" * 64,
        lockfile_sha256="beef",
        determinism={"n_threads": 1, "sort_kind": "quicksort"},
    )
    return RunRecord(
        context=context,
        model_name="hist_gradient_boosting",
        feature_set="engineered",
        n_features=119,
        train_seconds=4.0,
        splits=[{"name": "validation", "n_rows": 54284, "n_positive": 2563}],
        metrics={"validation": dict(VALIDATION_METRICS)},
        validation_reports=[],
        dataset={"row_order_sha256": "7174370c6f491542", "label": "default_within_60m"},
        mlflow_run_id=mlflow_run_id,
    )


def _config(tmp_path: Any) -> Config:
    return Config(
        raw={
            "seed": 42,
            "determinism": {"n_threads": 1},
            "tracking": {
                "enabled": True,
                "experiment_name": "test",
                "backend_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                "artifact_dir": str(tmp_path / "artifacts"),
            },
            "promotion": {
                "decision_split": "validation",
                "registered_model_name": "test-classifier",
                "production_alias": "production",
                "bootstrap_model": "baseline",
                "gates": {"min_improvement": {"metric": "average_precision", "min_delta": 0.01}},
            },
        },
        environment="production",
    )


def _report(promote: bool) -> GateReport:
    gate = GateResult(
        name="min_improvement",
        passed=promote,
        required=True,
        observed=0.6 if promote else 0.001,
        threshold=0.01,
        comparison="average_precision delta >= 0.010000",
        reason="" if promote else "improved by only 0.001000",
    )
    return GateReport(
        gates=[gate],
        candidate_name="hist_gradient_boosting",
        production_name="logistic_regression_baseline",
        production_source="bootstrap",
    )


class TestProductionIdentification:
    def test_an_absent_alias_yields_no_production(self, tmp_path: Any) -> None:
        """The first run of a fresh project has nothing registered."""
        assert registry.resolve_production(_config(tmp_path)) is None

    def test_the_newest_run_is_never_adopted_as_production(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that stops an accidental run becoming the incumbent.

        With no alias set, resolution must fall through to the configured
        bootstrap model rather than picking up any recent run.
        """
        config = _config(tmp_path)
        monkeypatch.setattr(compare, "resolve_production", lambda _c: None)

        trained: list[str] = []

        def _fake_training(**kwargs: Any) -> RunRecord:
            trained.append(kwargs["model_key"])
            record = _record()
            record.model_name = "logistic_regression_baseline"
            return record

        monkeypatch.setattr(compare, "run_training", _fake_training)
        result = compare.build_comparison(config, _record())

        assert trained == ["baseline"], "must train the configured bootstrap model"
        assert result.production.source == "bootstrap"

    def test_a_registered_alias_is_used_when_present(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path)
        monkeypatch.setattr(
            compare,
            "resolve_production",
            lambda _c: registry.ProductionModel(
                source="registry", name="test-classifier", version="3", run_id="prodrun"
            ),
        )
        monkeypatch.setattr(
            compare, "_metrics_from_registry", lambda *_a, **_k: dict(VALIDATION_METRICS)
        )
        result = compare.build_comparison(config, _record())
        assert result.production.source == "registry"
        assert result.production.version == "3"

    def test_the_production_description_says_where_it_came_from(self) -> None:
        registered = registry.ProductionModel(source="registry", name="m", version="2")
        bootstrap = registry.ProductionModel(source="bootstrap", name="baseline")
        assert "v2" in registered.describe()
        assert "bootstrap" in bootstrap.describe()

    def test_a_candidate_without_decision_metrics_is_refused(self, tmp_path: Any) -> None:
        record = _record()
        record.metrics = {"test": dict(VALIDATION_METRICS)}
        with pytest.raises(compare.ProductionResolutionError, match="cannot be judged"):
            compare.build_comparison(_config(tmp_path), record)


class TestRegistrationRefusesRejectedCandidates:
    def test_a_failed_report_cannot_be_registered(self, tmp_path: Any) -> None:
        """The safety property of the milestone, asserted at the registry door.

        Even if a caller wired the pipeline wrongly, registration itself refuses
        a candidate whose gates did not pass.
        """
        with pytest.raises(ValueError, match="refusing to register"):
            registry.register_candidate(_config(tmp_path), _record(), _report(promote=False))

    def test_the_refusal_names_the_failed_gates(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="min_improvement"):
            registry.register_candidate(_config(tmp_path), _record(), _report(promote=False))

    def test_a_candidate_without_an_mlflow_run_is_not_registered(self, tmp_path: Any) -> None:
        name, version = registry.register_candidate(
            _config(tmp_path), _record(mlflow_run_id=None), _report(promote=True)
        )
        assert (name, version) == (None, None)


class TestVersionTags:
    def test_traceability_to_both_run_identities(self) -> None:
        tags = registry._version_tags(_record(), _report(promote=True))
        assert tags["mlflow_run_id"] == "mlrun123"
        assert tags["platform_run_id"] == "candidate-20260907T000000Z"

    def test_provenance_is_carried(self) -> None:
        tags = registry._version_tags(_record(), _report(promote=True))
        assert tags["git_revision"] == "abc123"
        assert tags["config_fingerprint"] == "ffff0000"
        assert tags["dataset_sha256"] == "a" * 64
        assert tags["dataset_row_order_sha256"] == "7174370c6f491542"

    def test_the_deciding_metrics_are_carried(self) -> None:
        tags = registry._version_tags(_record(), _report(promote=True))
        assert tags["decision_split"] == "validation"
        assert tags["validation_average_precision"] == str(VALIDATION_METRICS["average_precision"])

    def test_the_incumbent_it_replaced_is_recorded(self) -> None:
        tags = registry._version_tags(_record(), _report(promote=True))
        assert tags["promoted_over"] == "logistic_regression_baseline"


class TestPromotionPipeline:
    """The pipeline must not register unless the gates passed."""

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, production: dict[str, Any]) -> list[str]:
        from ml_platform.pipelines import promote_pipeline

        registered: list[str] = []
        monkeypatch.setattr(
            promote_pipeline,
            "build_comparison",
            lambda _config, record, **_k: compare.Comparison(
                decision_split="validation",
                candidate=record,
                production_metrics=production,
                production=registry.ProductionModel(source="bootstrap", name="baseline"),
            ),
        )
        monkeypatch.setattr(
            promote_pipeline,
            "register_candidate",
            lambda *_a, **_k: (registered.append("called"), ("test-classifier", "1"))[1],
        )
        return registered

    def test_a_passing_candidate_is_registered_and_traceable(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ml_platform.pipelines.promote_pipeline import run_promotion

        registered = self._patch(monkeypatch, {**VALIDATION_METRICS, "average_precision": 0.10})
        config = _config(tmp_path)
        config.raw["promotion"]["gates"] = {
            "min_improvement": {"metric": "average_precision", "min_delta": 0.01}
        }
        config.raw["artifacts"] = {"benchmark_dir": str(tmp_path)}

        decision, _ = run_promotion(config=config, candidate=_record())
        assert decision.report.promote
        assert registered == ["called"]
        assert decision.registered
        assert decision.source_run_id == "mlrun123"

    def test_a_failing_candidate_is_never_registered(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected candidate keeps its run, but gets no registered version."""
        from ml_platform.pipelines.promote_pipeline import run_promotion

        # Production already matches the candidate, so the improvement gate fails.
        registered = self._patch(monkeypatch, dict(VALIDATION_METRICS))
        config = _config(tmp_path)
        config.raw["artifacts"] = {"benchmark_dir": str(tmp_path)}

        decision, _ = run_promotion(config=config, candidate=_record())
        assert not decision.report.promote
        assert registered == [], "registration must not be attempted"
        assert not decision.registered
        assert decision.version is None
        assert any("rejected" in note for note in decision.notes)

    def test_registration_can_be_skipped_for_a_dry_run(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ml_platform.pipelines.promote_pipeline import run_promotion

        registered = self._patch(monkeypatch, {**VALIDATION_METRICS, "average_precision": 0.10})
        config = _config(tmp_path)
        config.raw["artifacts"] = {"benchmark_dir": str(tmp_path)}

        decision, _ = run_promotion(config=config, candidate=_record(), register=False)
        assert decision.report.promote
        assert registered == []
        assert not decision.registered


class TestThresholdConfiguration:
    def test_thresholds_come_from_configuration(self, tmp_path: Any) -> None:
        config = _config(tmp_path)
        assert config.gate_config["min_improvement"]["min_delta"] == 0.01
        assert config.decision_split == "validation"

    def test_the_shipped_configuration_decides_on_validation(self) -> None:
        """A config that decided on test would invalidate the held-out evidence."""
        from ml_platform.config import load_config

        assert load_config("production").decision_split == "validation"

    def test_the_shipped_configuration_defines_every_gate(self) -> None:
        from ml_platform.config import load_config

        configured = set(load_config("production").gate_config)
        assert configured == {
            "min_improvement",
            "minimum_metric",
            "roc_auc_regression",
            "calibration",
            "recall_regression",
            "latency",
            "reproducibility",
        }
