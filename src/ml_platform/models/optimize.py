"""Hyperparameter optimisation with Optuna.

Three rules shape this module.

**The existing evaluation is the only scorer.** A trial is scored by
:func:`ml_platform.models.evaluate.evaluate_model`, the same function that
produces every other number in the project. There is no second metric
implementation to drift out of agreement with the first.

**Trials are scored on validation, never on test.** The test split is the final
honest number. Selecting hyperparameters against it would quietly turn it into a
training set, and the reported improvement would be optimistic. The objective
refuses to run against it.

**Sampling is seeded explicitly.** Optuna's default sampler seeds itself from
entropy, so an unseeded study is not reproducible. The seed comes from
configuration and is recorded with the study.

Optuna decides which trial is best, from values the project's own evaluation
produced. MLflow records that decision; it never makes it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ml_platform.models.evaluate import evaluate_model
from ml_platform.models.train import features_and_target, train

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna

    from ml_platform.config import Config
    from ml_platform.data.splitting import Split

LOGGER = logging.getLogger(__name__)

#: Metrics captured for every trial alongside the objective, so a study can be
#: read for calibration and operating-point behaviour rather than one number.
TRIAL_METRICS: tuple[str, ...] = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "brier_skill_score",
    "precision_at_capacity",
    "recall_at_capacity",
    "lift_at_capacity",
    "calibration_ratio",
)

#: Splits a search may be scored on. ``test`` is deliberately absent.
SCORABLE_SPLITS: frozenset[str] = frozenset({"train", "validation"})


class TrialSink(Protocol):
    """Anything that records a finished trial. Implemented by the MLflow tracker."""

    def log_trial(
        self,
        number: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        *,
        failed: bool = False,
    ) -> None: ...


@dataclass(frozen=True)
class TrialResult:
    """One finished trial, complete or failed."""

    number: int
    state: str
    params: dict[str, Any]
    value: float | None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.state == "COMPLETE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudyResult:
    """The outcome of a search, and the evidence for how it was reached."""

    study_name: str
    objective_metric: str
    objective_split: str
    direction: str
    sampler_seed: int
    n_trials: int
    trials: list[TrialResult]
    best_trial_number: int | None
    best_params: dict[str, Any]
    best_value: float | None
    baseline_value: float | None = None

    @property
    def n_completed(self) -> int:
        return sum(1 for t in self.trials if t.completed)

    @property
    def n_failed(self) -> int:
        return sum(1 for t in self.trials if not t.completed)

    @property
    def improvement_over_untuned(self) -> float | None:
        """Objective gain over the untuned candidate, on the same split."""
        if self.best_value is None or self.baseline_value is None:
            return None
        return round(self.best_value - self.baseline_value, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "objective_metric": self.objective_metric,
            "objective_split": self.objective_split,
            "direction": self.direction,
            "sampler_seed": self.sampler_seed,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
            "best_trial_number": self.best_trial_number,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "untuned_value": self.baseline_value,
            "improvement_over_untuned": self.improvement_over_untuned,
            "trials": [t.to_dict() for t in self.trials],
        }


class SearchSpaceError(ValueError):
    """Raised when the configured search space cannot be interpreted."""


def suggest_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Draw one point from the configured search space.

    The space lives in ``configs/model.yaml`` rather than in code, so widening a
    bound is a configuration change that shows up in the config fingerprint.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = str(spec.get("type", "")).lower()
        if kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise SearchSpaceError(f"parameter {name!r} has unsupported type {kind!r}")
    return params


class Objective:
    """Scores one hyperparameter point using the project's own evaluation."""

    def __init__(
        self,
        base_spec: dict[str, Any],
        splits: dict[str, Split],
        config: Config,
    ) -> None:
        split_name = config.objective_split
        if split_name not in SCORABLE_SPLITS:
            raise SearchSpaceError(
                f"objective split {split_name!r} is not one of {sorted(SCORABLE_SPLITS)}; "
                "the test split must not be used to select hyperparameters"
            )
        if split_name not in splits:
            raise SearchSpaceError(f"split {split_name!r} was not produced")

        self.base_spec = base_spec
        self.splits = splits
        self.config = config
        self.split_name = split_name
        self.metric = config.objective_metric
        self.search_space = config.search_space

    def build_spec(self, params: dict[str, Any]) -> dict[str, Any]:
        """The candidate specification with this trial's parameters applied."""
        merged = {**dict(self.base_spec.get("params") or {}), **params}
        return {**self.base_spec, "params": merged}

    def evaluate(self, params: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Fit and score one parameter set. Returns the objective and all metrics."""
        spec = self.build_spec(params)
        trained = train(spec, self.splits["train"], self.config.target_column)

        split = self.splits[self.split_name]
        features, target = features_and_target(
            split, trained.feature_set, self.config.target_column
        )
        result = evaluate_model(
            trained.pipeline,
            features,
            target,
            split=self.split_name,
            review_capacity=self.config.review_capacity,
        )

        metrics = {name: float(getattr(result, name)) for name in TRIAL_METRICS}
        metrics["train_seconds"] = trained.train_seconds
        return float(getattr(result, self.metric)), metrics

    def __call__(self, trial: optuna.Trial) -> float:
        params = suggest_params(trial, self.search_space)
        value, metrics = self.evaluate(params)
        # Carried on the trial so the logging callback can record the full metric
        # set, not just the single number Optuna optimises.
        trial.set_user_attr("metrics", metrics)
        return value


def _to_result(frozen: Any) -> TrialResult:
    """Convert an Optuna trial into a plain record."""
    metrics = dict(frozen.user_attrs.get("metrics") or {})
    return TrialResult(
        number=int(frozen.number),
        state=str(frozen.state.name),
        params=dict(frozen.params),
        value=None if frozen.value is None else float(frozen.value),
        metrics=metrics,
    )


def run_study(
    config: Config,
    splits: dict[str, Split],
    base_spec: dict[str, Any],
    *,
    study_name: str,
    n_trials: int | None = None,
    sink: TrialSink | None = None,
) -> StudyResult:
    """Run the search and return its result.

    A trial that raises is caught by Optuna, recorded in ``FAIL`` state and
    excluded from ``best_trial``. A failed trial can therefore never be selected
    as the best candidate, which is the safety property that matters here.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = Objective(base_spec, splits, config)
    budget = int(n_trials if n_trials is not None else config.n_trials)

    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        direction=config.objective_direction,
        sampler=sampler,
    )

    def _record(_study: Any, frozen: Any) -> None:
        """Log every finished trial, complete or failed."""
        result = _to_result(frozen)
        LOGGER.info(
            "trial %3d %-8s %s=%s",
            result.number,
            result.state,
            objective.metric,
            "n/a" if result.value is None else f"{result.value:.6f}",
        )
        if sink is not None:
            sink.log_trial(
                result.number,
                result.params,
                result.metrics if result.completed else {},
                failed=not result.completed,
            )

    study.optimize(objective, n_trials=budget, callbacks=[_record], catch=(Exception,))

    trials = [_to_result(t) for t in study.trials]
    completed = [t for t in trials if t.completed]

    best_number: int | None = None
    best_params: dict[str, Any] = {}
    best_value: float | None = None
    if completed:
        best = study.best_trial
        best_number = int(best.number)
        best_params = dict(best.params)
        best_value = None if best.value is None else float(best.value)

    # The untuned candidate, scored on the same split by the same code, so the
    # study can say whether searching achieved anything at all.
    untuned_value, _ = objective.evaluate({})

    return StudyResult(
        study_name=study_name,
        objective_metric=objective.metric,
        objective_split=objective.split_name,
        direction=config.objective_direction,
        sampler_seed=config.sampler_seed,
        n_trials=budget,
        trials=trials,
        best_trial_number=best_number,
        best_params=best_params,
        best_value=best_value,
        baseline_value=untuned_value,
    )
