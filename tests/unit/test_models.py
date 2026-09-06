"""Tests for model construction and fitting.

The property that matters most here is that preprocessing lives inside the
pipeline. If imputation or encoding were fitted outside it, the served model
would see differently-prepared features from the trained one, which is the most
common and least visible cause of a model that works in testing and fails in
production.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_platform.config import DateWindow
from ml_platform.data.splitting import Split
from ml_platform.models import baseline as mb
from ml_platform.models import train as mt

LOGREG_SPEC: dict[str, Any] = {
    "name": "test_logreg",
    "estimator": "sklearn.linear_model.LogisticRegression",
    "params": {"max_iter": 200, "random_state": 42},
    "feature_set": "core",
}
HGB_SPEC: dict[str, Any] = {
    "name": "test_hgb",
    "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
    "params": {"max_iter": 20, "random_state": 42},
    "feature_set": "engineered",
}


def _training_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    terms = rng.choice([60, 84, 120, 240], size=n)
    gross = rng.integers(20_000, 500_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "ApprovalDate": pd.to_datetime("2002-01-01")
            + pd.to_timedelta(rng.integers(0, 700, size=n), unit="D"),
            "DisbursementDate": pd.to_datetime("2002-03-01")
            + pd.to_timedelta(rng.integers(0, 700, size=n), unit="D"),
            "Term": terms.astype(float),
            "NoEmp": rng.integers(0, 40, size=n).astype(float),
            "CreateJob": rng.integers(0, 8, size=n).astype(float),
            "RetainedJob": rng.integers(0, 15, size=n).astype(float),
            "GrAppv": gross,
            "SBA_Appv": gross * 0.75,
            "DisbursementGross": gross * 0.95,
            "State": rng.choice(["CA", "TX", "NY"], size=n),
            "BankState": rng.choice(["CA", "TX", "NY"], size=n),
            "RevLineCr": rng.choice(["Y", "N", "UNK"], size=n),
            "LowDoc": rng.choice(["Y", "N"], size=n),
            "UrbanRural": rng.choice([0.0, 1.0, 2.0], size=n),
            "NewExist": rng.choice([1.0, 2.0], size=n),
            "NAICS": rng.choice(["722410", "451120", "0"], size=n),
            "FranchiseCode": rng.choice([0.0, 44321.0], size=n),
            # Short terms carry more risk, so there is a signal to learn.
            "target": ((terms < 100) & (rng.random(n) < 0.6)).astype(int),
        }
    )


def _split(frame: pd.DataFrame, name: str = "train") -> Split:
    return Split(name, DateWindow.from_mapping({"start": "2000-01-01", "end": "2005-12-31"}), frame)


class TestEstimatorResolution:
    def test_a_dotted_path_resolves_to_the_class(self) -> None:
        assert mb._import_estimator("sklearn.linear_model.LogisticRegression") is LogisticRegression

    def test_an_unknown_module_is_an_import_error(self) -> None:
        with pytest.raises(ImportError):
            mb._import_estimator("sklearn.not_a_module.Thing")

    def test_a_non_class_target_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="not a class"):
            mb._import_estimator("sklearn.metrics.roc_auc_score")


class TestPreprocessor:
    @pytest.mark.parametrize("feature_set", ["core", "engineered"])
    def test_it_covers_every_declared_column(self, feature_set: str) -> None:
        from ml_platform.features.engineering import feature_columns

        numeric, categorical = feature_columns(feature_set)
        preprocessor = mb.build_preprocessor(feature_set)
        covered = {c for _, _, columns in preprocessor.transformers for c in columns}
        assert covered == set(numeric) | set(categorical)

    def test_gradient_boosting_gets_a_dense_encoder(self) -> None:
        """HistGradientBoosting cannot consume a sparse matrix."""
        pipeline = mb.build_model(HGB_SPEC)
        encoder = pipeline.named_steps["preprocess"].transformers[1][1].named_steps["encode"]
        assert encoder.sparse_output is False
        assert isinstance(pipeline.named_steps["estimator"], HistGradientBoostingClassifier)

    def test_linear_models_keep_a_sparse_encoder(self) -> None:
        """One-hot state and sector are wide; sparsity is worth keeping where it works."""
        pipeline = mb.build_model(LOGREG_SPEC)
        encoder = pipeline.named_steps["preprocess"].transformers[1][1].named_steps["encode"]
        assert encoder.sparse_output is True

    def test_gradient_boosting_fits_end_to_end(self) -> None:
        pipeline = mb.build_model(HGB_SPEC)
        frame = _training_frame()
        pipeline.fit(*mt.features_and_target(_split(frame), "engineered", "target"))
        features, _ = mt.features_and_target(_split(frame), "engineered", "target")
        assert pipeline.predict_proba(features).shape == (len(frame), 2)

    def test_unseen_categories_are_absorbed_not_rejected(self) -> None:
        """`handle_unknown` must be set, or a new state crashes inference."""
        pipeline = mb.build_model(LOGREG_SPEC)
        encoder = pipeline.named_steps["preprocess"].transformers[1][1].named_steps["encode"]
        assert encoder.handle_unknown == "infrequent_if_exist"


class TestBuildModel:
    def test_a_model_is_a_pipeline_with_preprocessing_inside(self) -> None:
        """Preprocessing must travel with the model, not sit beside it."""
        pipeline = mb.build_model(LOGREG_SPEC)
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["preprocess", "estimator"]

    def test_configured_parameters_reach_the_estimator(self) -> None:
        pipeline = mb.build_model(LOGREG_SPEC)
        assert pipeline.named_steps["estimator"].max_iter == 200

    def test_candidate_overrides_are_applied(self) -> None:
        config = {"candidate": HGB_SPEC}
        pipeline = mb.build_candidate(config, {"max_iter": 7})
        assert pipeline.named_steps["estimator"].max_iter == 7

    def test_overrides_do_not_mutate_the_configuration(self) -> None:
        config = {"candidate": dict(HGB_SPEC)}
        mb.build_candidate(config, {"max_iter": 7})
        assert config["candidate"]["params"]["max_iter"] == 20

    def test_baseline_builds_from_the_configuration_block(self) -> None:
        pipeline = mb.build_baseline({"baseline": LOGREG_SPEC})
        assert isinstance(pipeline.named_steps["estimator"], LogisticRegression)


class TestTraining:
    def test_a_fitted_model_reports_what_it_saw(self) -> None:
        frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(frame), "target")
        assert trained.n_train_rows == len(frame)
        assert trained.feature_set == "core"
        assert trained.train_seconds >= 0

    def test_a_single_class_split_is_refused(self) -> None:
        """Fitting on one class produces a model that cannot be evaluated."""
        frame = _training_frame()
        frame["target"] = 0
        with pytest.raises(ValueError, match="single class"):
            mt.train(LOGREG_SPEC, _split(frame), "target")

    def test_training_is_repeatable(self) -> None:
        frame = _training_frame()
        first = mt.train(HGB_SPEC, _split(frame), "target")
        second = mt.train(HGB_SPEC, _split(frame), "target")
        features, _ = mt.features_and_target(_split(frame), "engineered", "target")
        assert np.array_equal(
            first.pipeline.predict_proba(features), second.pipeline.predict_proba(features)
        )

    def test_probabilities_are_valid(self) -> None:
        frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(frame), "target")
        features, _ = mt.features_and_target(_split(frame), "core", "target")
        scores = trained.pipeline.predict_proba(features)[:, 1]
        assert ((scores >= 0) & (scores <= 1)).all()


class TestServingRobustness:
    """A served model meets values training never saw."""

    def test_an_unseen_category_does_not_crash_inference(self) -> None:
        """A state absent from training must not break a prediction."""
        train_frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(train_frame), "target")

        serve_frame = _training_frame(n=20, seed=99)
        serve_frame["State"] = "ZZ"
        features, _ = mt.features_and_target(_split(serve_frame), "core", "target")
        assert len(trained.pipeline.predict_proba(features)) == 20

    def test_missing_numeric_values_are_imputed_at_inference(self) -> None:
        train_frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(train_frame), "target")

        serve_frame = _training_frame(n=10, seed=5)
        serve_frame.loc[:, "NoEmp"] = np.nan
        features, _ = mt.features_and_target(_split(serve_frame), "core", "target")
        scores = trained.pipeline.predict_proba(features)[:, 1]
        assert np.isfinite(scores).all()

    def test_imputation_uses_training_statistics_only(self) -> None:
        """The imputer must be fitted on training data, not refitted at serve time."""
        train_frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(train_frame), "target")
        imputer = trained.pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        expected = train_frame["Term"].median()
        assert imputer.named_steps["impute"].statistics_[0] == pytest.approx(expected)


class TestPersistence:
    def test_a_saved_model_round_trips(self, tmp_path: Any) -> None:
        frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(frame), "target")
        path = trained.save(tmp_path / "model.joblib")
        assert path.exists()

        loaded = mt.load_model(path)
        assert loaded.name == trained.name
        assert loaded.feature_set == trained.feature_set

    def test_a_reloaded_model_predicts_identically(self, tmp_path: Any) -> None:
        """Serialisation must not change a single prediction."""
        frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(frame), "target")
        loaded = mt.load_model(trained.save(tmp_path / "model.joblib"))

        features, _ = mt.features_and_target(_split(frame), "core", "target")
        assert np.array_equal(
            trained.pipeline.predict_proba(features), loaded.pipeline.predict_proba(features)
        )

    def test_saving_creates_missing_directories(self, tmp_path: Any) -> None:
        frame = _training_frame()
        trained = mt.train(LOGREG_SPEC, _split(frame), "target")
        assert trained.save(tmp_path / "a" / "b" / "model.joblib").exists()
