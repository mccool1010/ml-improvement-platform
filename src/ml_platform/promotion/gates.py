"""Quality gates.

A candidate is promoted only if it passes every mandatory gate. Any failure
rejects it and leaves production untouched.

Two rules govern this module.

**Decisions use the validation split.** The test split is held-out evidence. A
threshold chosen against it, or a promotion justified by it, converts it into a
training signal and it stops being evidence for anything. The evaluator refuses
to read a test split.

**Gates are pure.** Everything here operates on two metric dictionaries and a
small context. There is no MLflow, no filesystem and no model. That makes every
failure path testable in milliseconds, which matters because the failure paths
are the ones that protect production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: The split a promotion decision may read. Anything else is refused.
DECISION_SPLIT = "validation"

#: Metrics where a smaller number is better, so "no regression" inverts.
LOWER_IS_BETTER = frozenset({"brier_score"})


class GateConfigurationError(ValueError):
    """Raised when the configured gates cannot be evaluated as written."""


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, with the numbers behind it."""

    name: str
    passed: bool
    required: bool
    observed: float | None
    threshold: float | None
    comparison: str
    reason: str = ""

    @property
    def blocking(self) -> bool:
        """A required gate that failed. These are what stop a promotion."""
        return self.required and not self.passed

    def describe(self) -> str:
        state = "PASS" if self.passed else ("FAIL" if self.required else "WARN")
        observed = "n/a" if self.observed is None else f"{self.observed:.6f}"
        return f"[{state}] {self.name}: observed {observed}, needs {self.comparison}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateReport:
    """The full promotion verdict."""

    gates: list[GateResult]
    decision_split: str = DECISION_SPLIT
    candidate_name: str = ""
    production_name: str = ""
    production_source: str = ""

    @property
    def promote(self) -> bool:
        """True only when every mandatory gate passed."""
        return not any(gate.blocking for gate in self.gates)

    @property
    def failures(self) -> list[GateResult]:
        return [gate for gate in self.gates if gate.blocking]

    @property
    def warnings(self) -> list[GateResult]:
        return [gate for gate in self.gates if not gate.passed and not gate.required]

    def summary(self) -> str:
        verdict = "PROMOTE" if self.promote else "REJECT"
        passed = sum(1 for gate in self.gates if gate.passed)
        return (
            f"[{verdict}] {passed}/{len(self.gates)} gates passed on the "
            f"{self.decision_split} split, {len(self.failures)} blocking failure(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "decision_split": self.decision_split,
            "candidate": self.candidate_name,
            "production": self.production_name,
            "production_source": self.production_source,
            "n_gates": len(self.gates),
            "n_passed": sum(1 for gate in self.gates if gate.passed),
            "blocking_failures": [gate.name for gate in self.failures],
            "gates": [gate.to_dict() for gate in self.gates],
        }


def _required(spec: dict[str, Any]) -> bool:
    return bool(spec.get("required", True))


def _metric(metrics: dict[str, Any], name: str, owner: str) -> float:
    if name not in metrics:
        raise GateConfigurationError(f"{owner} metrics have no {name!r}")
    return float(metrics[name])


def gate_min_improvement(
    candidate: dict[str, Any], production: dict[str, Any], spec: dict[str, Any]
) -> GateResult:
    """The candidate must beat production by a stated margin, not merely tie."""
    metric = str(spec.get("metric", "average_precision"))
    minimum = float(spec.get("min_delta", 0.01))
    delta = _metric(candidate, metric, "candidate") - _metric(production, metric, "production")
    passed = delta >= minimum
    return GateResult(
        name="min_improvement",
        passed=passed,
        required=_required(spec),
        observed=round(delta, 6),
        threshold=minimum,
        comparison=f"{metric} delta >= {minimum:.6f}",
        reason=(
            ""
            if passed
            else f"{metric} improved by only {delta:.6f}, below the required {minimum:.6f}"
        ),
    )


def gate_minimum_metric(candidate: dict[str, Any], spec: dict[str, Any]) -> GateResult:
    """An absolute floor, so a candidate cannot win by beating a poor incumbent."""
    metric = str(spec.get("metric", "average_precision"))
    minimum = float(spec.get("minimum", 0.0))
    observed = _metric(candidate, metric, "candidate")
    passed = observed >= minimum
    return GateResult(
        name="minimum_metric",
        passed=passed,
        required=_required(spec),
        observed=round(observed, 6),
        threshold=minimum,
        comparison=f"{metric} >= {minimum:.6f}",
        reason="" if passed else f"{metric} of {observed:.6f} is below the floor {minimum:.6f}",
    )


def gate_no_regression(
    candidate: dict[str, Any],
    production: dict[str, Any],
    spec: dict[str, Any],
    *,
    name: str,
) -> GateResult:
    """Guard a metric against sliding backwards while another one improves."""
    metric = str(spec["metric"])
    tolerance = float(spec.get("max_decline", 0.0))
    candidate_value = _metric(candidate, metric, "candidate")
    production_value = _metric(production, metric, "production")

    if metric in LOWER_IS_BETTER:
        decline = candidate_value - production_value
        comparison = f"{metric} <= production + {tolerance:.6f}"
    else:
        decline = production_value - candidate_value
        comparison = f"{metric} >= production - {tolerance:.6f}"

    passed = decline <= tolerance
    return GateResult(
        name=name,
        passed=passed,
        required=_required(spec),
        observed=round(candidate_value, 6),
        threshold=round(production_value, 6),
        comparison=comparison,
        reason=(
            ""
            if passed
            else (
                f"{metric} regressed by {decline:.6f} against production "
                f"{production_value:.6f}, beyond the tolerance {tolerance:.6f}"
            )
        ),
    )


def gate_calibration(
    candidate: dict[str, Any], production: dict[str, Any], spec: dict[str, Any]
) -> GateResult:
    """Probabilities must stay honest, not just the ranking.

    Under drift a model can hold its ranking while its probabilities go badly
    wrong. A gate watching only ranking would wave that through, which is why
    calibration is checked separately and against two conditions: no worse than
    production, and better than always predicting the base rate.
    """
    tolerance = float(spec.get("max_decline", 0.0))
    min_skill = float(spec.get("min_skill", 0.0))

    candidate_brier = _metric(candidate, "brier_score", "candidate")
    production_brier = _metric(production, "brier_score", "production")
    skill = _metric(candidate, "brier_skill_score", "candidate")

    regressed = (candidate_brier - production_brier) > tolerance
    unskilled = skill <= min_skill
    passed = not regressed and not unskilled

    reasons: list[str] = []
    if regressed:
        reasons.append(
            f"Brier score {candidate_brier:.6f} is worse than production {production_brier:.6f}"
        )
    if unskilled:
        reasons.append(
            f"Brier skill {skill:.6f} is not above {min_skill:.6f}; the model is no better "
            "calibrated than always predicting the base rate"
        )

    return GateResult(
        name="calibration",
        passed=passed,
        required=_required(spec),
        observed=round(candidate_brier, 6),
        threshold=round(production_brier + tolerance, 6),
        comparison=f"brier <= production + {tolerance:.6f} and skill > {min_skill:.6f}",
        reason="; ".join(reasons),
    )


def mean_prediction_ms(metrics: dict[str, Any]) -> float | None:
    """Mean time to score one row, in milliseconds.

    Derived from the evaluation pass, so it measures batch throughput rather
    than single-request latency. It is a coarse proxy: a true p99 needs the
    serving path, which does not exist yet. It is still worth gating, because a
    model that is orders of magnitude slower than its predecessor shows up here.
    """
    seconds = metrics.get("inference_seconds")
    rows = metrics.get("n_rows")
    if seconds is None or not rows:
        return None
    return 1000.0 * float(seconds) / float(rows)


def gate_latency(candidate: dict[str, Any], spec: dict[str, Any]) -> GateResult:
    """An accurate model that cannot be served fast enough cannot be used."""
    budget = float(spec.get("max_mean_prediction_ms", 1.0))
    observed = mean_prediction_ms(candidate)

    if observed is None:
        return GateResult(
            name="latency",
            passed=False,
            required=_required(spec),
            observed=None,
            threshold=budget,
            comparison=f"mean prediction <= {budget:.4f} ms",
            reason="the candidate carries no timing measurement, so latency cannot be checked",
        )

    passed = observed <= budget
    return GateResult(
        name="latency",
        passed=passed,
        required=_required(spec),
        observed=round(observed, 6),
        threshold=budget,
        comparison=f"mean prediction <= {budget:.4f} ms",
        reason=(
            ""
            if passed
            else f"mean prediction time {observed:.4f} ms exceeds the budget {budget:.4f} ms"
        ),
    )


def gate_reproducibility(context: dict[str, Any], spec: dict[str, Any]) -> GateResult:
    """A result that cannot be reproduced from its provenance is not evidence."""
    require_clean = bool(spec.get("require_clean_revision", True))
    reproducible = bool(context.get("reproducible", False))
    passed = reproducible or not require_clean

    reasons: list[str] = []
    if not reproducible:
        if context.get("git_revision", "unknown") == "unknown":
            reasons.append("the code version is unknown")
        if context.get("git_dirty"):
            reasons.append("the working tree had uncommitted changes")
        if context.get("lockfile_sha256", "absent") == "absent":
            reasons.append("no dependency lockfile was recorded")
        if not reasons:
            reasons.append("the run is not reproducible from its recorded provenance")

    return GateResult(
        name="reproducibility",
        passed=passed,
        required=_required(spec),
        observed=None,
        threshold=None,
        comparison="clean revision, and a recorded lockfile",
        reason="" if passed else "; ".join(reasons),
    )


#: Gate name to the builder that evaluates it. Configuration selects which run.
GATE_BUILDERS: tuple[str, ...] = (
    "min_improvement",
    "minimum_metric",
    "roc_auc_regression",
    "calibration",
    "recall_regression",
    "latency",
    "reproducibility",
)


def evaluate_gates(
    candidate_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
    gate_config: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    split: str = DECISION_SPLIT,
    candidate_name: str = "",
    production_name: str = "",
    production_source: str = "",
) -> GateReport:
    """Run every configured gate and return the verdict.

    ``split`` must be the validation split. Passing the test split raises,
    because a promotion justified by held-out evidence destroys that evidence.
    """
    if split != DECISION_SPLIT:
        raise GateConfigurationError(
            f"promotion must be decided on the {DECISION_SPLIT!r} split, not {split!r}; "
            "the test split is held-out evidence and must not inform the decision"
        )

    ctx = dict(context or {})
    results: list[GateResult] = []

    if "min_improvement" in gate_config:
        results.append(
            gate_min_improvement(
                candidate_metrics, production_metrics, gate_config["min_improvement"]
            )
        )
    if "minimum_metric" in gate_config:
        results.append(gate_minimum_metric(candidate_metrics, gate_config["minimum_metric"]))
    if "roc_auc_regression" in gate_config:
        results.append(
            gate_no_regression(
                candidate_metrics,
                production_metrics,
                gate_config["roc_auc_regression"],
                name="roc_auc_regression",
            )
        )
    if "calibration" in gate_config:
        results.append(
            gate_calibration(candidate_metrics, production_metrics, gate_config["calibration"])
        )
    if "recall_regression" in gate_config:
        results.append(
            gate_no_regression(
                candidate_metrics,
                production_metrics,
                gate_config["recall_regression"],
                name="recall_regression",
            )
        )
    if "latency" in gate_config:
        results.append(gate_latency(candidate_metrics, gate_config["latency"]))
    if "reproducibility" in gate_config:
        results.append(gate_reproducibility(ctx, gate_config["reproducibility"]))

    if not results:
        raise GateConfigurationError("no gates are configured; refusing to promote unchecked")

    return GateReport(
        gates=results,
        decision_split=split,
        candidate_name=candidate_name,
        production_name=production_name,
        production_source=production_source,
    )


@dataclass
class PromotionDecision:
    """A gate report plus what was done about it."""

    report: GateReport
    registered: bool = False
    registered_model: str | None = None
    version: str | None = None
    source_run_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "registered": self.registered,
            "registered_model": self.registered_model,
            "version": self.version,
            "source_run_id": self.source_run_id,
            "notes": self.notes,
        }
