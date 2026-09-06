"""Tests for hyperparameter optimisation.

Three properties carry the milestone and are tested directly:

* the objective is scored by the project's existing evaluation, on the
  validation split, and **never** on test;
* a trial that raises is recorded as failed and can never be selected as best;
* sampling is seeded, so a study is reproducible.

The tests use a tiny in-memory dataset and a handful of trials. Running the real
budget here would cost minutes per test run and prove nothing extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_platform.config import Config, DateWindow
from ml_platform.data.splitting import Split
from ml_platform.models import optimize

SPEC: dict[str, Any] = {
    "name": "hist_gradient_boosting",
    "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
    "params": {"learning_rate": 0.1, "max_iter": 20, "random_state": 42},
    "feature_set": "core",
}

SEARCH_SPACE: dict[str, Any] = {
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
    "max_leaf_nodes": {"type": "int", "low": 8, "high": 24},
}


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    terms = rng.choice([60, 84, 120, 240], size=n).astype(float)
    gross = rng.integers(20_000, 400_000, size=n).astype(float)
    return pd.DataFrame(
        {
            "ApprovalDate": pd.to_datetime("2002-01-01")
            + pd.to_timedelta(rng.integers(0, 500, size=n), unit="D"),
            "DisbursementDate": pd.to_datetime("2002-03-01")
            + pd.to_timedelta(rng.integers(0, 500, size=n), unit="D"),
            "Term": terms,
            "NoEmp": rng.integers(1, 40, size=n).astype(float),
            "CreateJob": rng.integers(0, 8, size=n).astype(float),
            "RetainedJob": rng.integers(0, 12, size=n).astype(float),
            "GrAppv": gross,
            "SBA_Appv": gross * 0.75,
            "DisbursementGross": gross * 0.95,
            "State": rng.choice(["CA", "TX", "NY"], size=n),
            "BankState": rng.choice(["CA", "TX", "NY"], size=n),
            "RevLineCr": rng.choice(["Y", "N"], size=n),
            "LowDoc": rng.choice(["Y", "N"], size=n),
            "UrbanRural": rng.choice([0.0, 1.0, 2.0], size=n),
            "NewExist": rng.choice([1.0, 2.0], size=n),
            "NAICS": rng.choice(["722410", "451120"], size=n),
            "FranchiseCode": rng.choice([0.0, 44321.0], size=n),
            "target": ((terms < 100) & (rng.random(n) < 0.7)).astype(int),
        }
    )


def _splits() -> dict[str, Split]:
    window = DateWindow.from_mapping({"start": "2000-01-01", "end": "2005-12-31"})
    return {
        "train": Split("train", window, _frame(seed=0)),
        "validation": Split("validation", window, _frame(seed=1)),
        "test": Split("test", window, _frame(seed=2)),
    }


def _config(**overrides: Any) -> Config:
    objective = {"metric": "average_precision", "split": "validation", "direction": "maximize"}
    objective.update(overrides.pop("objective", {}))
    return Config(
        raw={
            "seed": 42,
            "determinism": {"n_threads": 1},
            "label": {"target_column": "target"},
            "evaluation": {"review_capacity": 0.10},
            "optimization": {
                "n_trials": 3,
                "sampler_seed": 42,
                "objective": objective,
                "search_space": SEARCH_SPACE,
                **overrides,
            },
        },
        environment="test",
    )


class TestSearchSpace:
    def test_the_space_comes_from_configuration(self) -> None:
        """Widening a bound must be a config change, visible in the fingerprint."""
        assert set(_config().search_space) == {"learning_rate", "max_leaf_nodes"}

    def test_every_suggested_value_is_inside_its_bounds(self) -> None:
        import optuna

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        for _ in range(8):
            trial = study.ask()
            params = optimize.suggest_params(trial, SEARCH_SPACE)
            assert 0.05 <= params["learning_rate"] <= 0.3
            assert 8 <= params["max_leaf_nodes"] <= 24
            study.tell(trial, 0.5)

    def test_integer_parameters_stay_integers(self) -> None:
        import optuna

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
        params = optimize.suggest_params(study.ask(), SEARCH_SPACE)
        assert isinstance(params["max_leaf_nodes"], int)

    def test_an_unsupported_parameter_type_is_rejected(self) -> None:
        import optuna

        study = optuna.create_study()
        with pytest.raises(optimize.SearchSpaceError, match="unsupported type"):
            optimize.suggest_params(study.ask(), {"x": {"type": "matrix", "low": 1, "high": 2}})


class TestObjective:
    def test_it_scores_on_the_configured_split(self) -> None:
        objective = optimize.Objective(SPEC, _splits(), _config())
        assert objective.split_name == "validation"

    def test_scoring_on_the_test_split_is_refused(self) -> None:
        """Selecting hyperparameters on test would destroy the final honest number."""
        with pytest.raises(optimize.SearchSpaceError, match="must not be used"):
            optimize.Objective(SPEC, _splits(), _config(objective={"split": "test"}))

    def test_an_absent_split_is_reported(self) -> None:
        with pytest.raises(optimize.SearchSpaceError, match="was not produced"):
            optimize.Objective(SPEC, {"train": _splits()["train"]}, _config())

    def test_overrides_are_merged_onto_the_configured_spec(self) -> None:
        objective = optimize.Objective(SPEC, _splits(), _config())
        spec = objective.build_spec({"learning_rate": 0.25})
        assert spec["params"]["learning_rate"] == 0.25
        # Untouched parameters survive.
        assert spec["params"]["random_state"] == 42
        assert spec["name"] == "hist_gradient_boosting"

    def test_the_base_specification_is_not_mutated(self) -> None:
        objective = optimize.Objective(SPEC, _splits(), _config())
        objective.build_spec({"learning_rate": 0.99})
        assert SPEC["params"]["learning_rate"] == 0.1

    def test_evaluation_returns_the_objective_and_the_full_metric_set(self) -> None:
        """The single number Optuna optimises comes from the shared evaluator."""
        objective = optimize.Objective(SPEC, _splits(), _config())
        value, metrics = objective.evaluate({"max_leaf_nodes": 12})
        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(metrics["average_precision"])
        assert {"roc_auc", "brier_score", "calibration_ratio"} <= set(metrics)


@pytest.fixture(scope="module")
def result() -> optimize.StudyResult:
    """One small study, shared by the assertions below."""
    return optimize.run_study(_config(), _splits(), SPEC, study_name="unit-study", n_trials=4)


class TestStudy:
    def test_every_trial_is_recorded(self, result: optimize.StudyResult) -> None:
        assert len(result.trials) == 4
        assert result.n_completed == 4
        assert result.n_failed == 0

    def test_the_best_trial_is_reported(self, result: optimize.StudyResult) -> None:
        assert result.best_trial_number is not None
        assert result.best_params
        assert result.best_value is not None

    def test_the_best_value_is_the_maximum_of_completed_trials(
        self, result: optimize.StudyResult
    ) -> None:
        """Optuna selects the winner; this asserts it selected the right one."""
        values = [t.value for t in result.trials if t.completed and t.value is not None]
        assert result.best_value == pytest.approx(max(values))

    def test_the_untuned_candidate_is_scored_for_comparison(
        self, result: optimize.StudyResult
    ) -> None:
        assert result.baseline_value is not None
        assert result.improvement_over_untuned is not None

    def test_the_objective_definition_is_recorded(self, result: optimize.StudyResult) -> None:
        assert result.objective_metric == "average_precision"
        assert result.objective_split == "validation"
        assert result.sampler_seed == 42

    def test_the_result_serialises(self, result: optimize.StudyResult) -> None:
        payload = result.to_dict()
        assert payload["n_completed"] == 4
        assert len(payload["trials"]) == 4


class TestFailedTrials:
    """A failed trial must never be selected as the best candidate."""

    @staticmethod
    def _explosive(fail_on: set[int]) -> Any:
        """An objective that raises for the given trial numbers."""

        def objective(trial: Any) -> float:
            params = optimize.suggest_params(trial, SEARCH_SPACE)
            if trial.number in fail_on:
                raise RuntimeError(f"trial {trial.number} exploded")
            trial.set_user_attr("metrics", {"average_precision": params["learning_rate"]})
            return float(params["learning_rate"])

        return objective

    def _run(self, fail_on: set[int], monkeypatch: pytest.MonkeyPatch) -> optimize.StudyResult:
        monkeypatch.setattr(
            optimize, "Objective", lambda *_a, **_k: _StubObjective(self._explosive(fail_on))
        )
        return optimize.run_study(_config(), _splits(), SPEC, study_name="fail-study", n_trials=5)

    def test_a_raising_trial_is_recorded_as_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run({1, 3}, monkeypatch)
        assert result.n_failed == 2
        assert result.n_completed == 3

    def test_a_failed_trial_carries_no_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run({1}, monkeypatch)
        failed = [t for t in result.trials if not t.completed]
        assert all(t.value is None for t in failed)

    def test_the_best_trial_is_never_a_failed_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run({0, 2, 4}, monkeypatch)
        failed_numbers = {t.number for t in result.trials if not t.completed}
        assert result.best_trial_number not in failed_numbers

    def test_a_search_survives_failures_and_still_returns_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run({0, 1, 2, 3}, monkeypatch)
        assert result.n_completed == 1
        assert result.best_params


class _StubObjective:
    """Wraps a bare callable so run_study's Objective contract is satisfied."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.metric = "average_precision"
        self.split_name = "validation"

    def __call__(self, trial: Any) -> float:
        return float(self._fn(trial))

    def evaluate(self, _params: dict[str, Any]) -> tuple[float, dict[str, float]]:
        return 0.0, {}


class TestDeterminism:
    def test_two_studies_with_the_same_seed_explore_the_same_points(self) -> None:
        """Optuna seeds its sampler from entropy unless told otherwise."""
        first = optimize.run_study(_config(), _splits(), SPEC, study_name="a", n_trials=4)
        second = optimize.run_study(_config(), _splits(), SPEC, study_name="b", n_trials=4)
        assert [t.params for t in first.trials] == [t.params for t in second.trials]

    def test_the_same_seed_reaches_the_same_best_trial(self) -> None:
        first = optimize.run_study(_config(), _splits(), SPEC, study_name="a", n_trials=4)
        second = optimize.run_study(_config(), _splits(), SPEC, study_name="b", n_trials=4)
        assert first.best_trial_number == second.best_trial_number
        assert first.best_value == pytest.approx(second.best_value)

    def test_a_different_seed_explores_different_points(self) -> None:
        first = optimize.run_study(_config(), _splits(), SPEC, study_name="a", n_trials=4)
        other = optimize.run_study(
            _config(sampler_seed=7), _splits(), SPEC, study_name="b", n_trials=4
        )
        assert [t.params for t in first.trials] != [t.params for t in other.trials]


class TestTrialSink:
    def test_every_trial_is_offered_to_the_sink(self) -> None:
        """The MLflow tracker receives failures too, so a study run list is honest."""
        recorded: list[tuple[int, bool]] = []

        class _Sink:
            def log_trial(
                self,
                number: int,
                params: dict[str, Any],  # noqa: ARG002 - protocol signature
                metrics: dict[str, float],  # noqa: ARG002 - protocol signature
                *,
                failed: bool = False,
            ) -> None:
                recorded.append((number, failed))

        optimize.run_study(_config(), _splits(), SPEC, study_name="sink", n_trials=3, sink=_Sink())
        assert [n for n, _ in recorded] == [0, 1, 2]
        assert not any(failed for _, failed in recorded)
