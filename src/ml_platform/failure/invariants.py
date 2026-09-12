"""The properties that must hold while things are breaking.

An invariant here is deliberately narrow and checkable against the real system.
"The platform is resilient" is not an invariant; "a request never receives a
fabricated prediction" is, because there is a specific observation that would
falsify it.

The distinction that matters throughout: **failing closed is success.** A
service that answers 503 because its model tier is gone has behaved correctly. A
service that answers 200 with a number it invented has not, and would be far
harder to notice. Several invariants below are therefore satisfied by errors and
violated by successes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Named invariants, so a report says which property a scenario actually proved
#: rather than leaving a reader to infer it.
NO_FABRICATED_PREDICTIONS = "no_fabricated_predictions"
PRODUCTION_ALIAS_UNCHANGED = "production_alias_unchanged"
FAILED_CANDIDATE_NOT_PRODUCTION = "failed_candidate_not_production"
CANARY_ROLLBACK_RESTORES_INCUMBENT = "canary_rollback_restores_incumbent"
TELEMETRY_FAILURE_ISOLATED = "telemetry_failure_isolated"
RECOVERY_RETURNS_KNOWN_STATE = "recovery_returns_known_state"
DEPENDENCY_FAILURE_IS_EXPLICIT = "dependency_failure_is_explicit"
INFERENCE_UNAFFECTED = "inference_unaffected"


@dataclass
class InvariantResult:
    """Whether one invariant held, and the observation that decided it."""

    name: str
    held: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        mark = "HELD" if self.held else "VIOLATED"
        return f"[{mark}] {self.name}: {self.detail}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def no_fabricated_predictions(responses: list[dict[str, Any]]) -> InvariantResult:
    """No request succeeded with a score while the model tier was unavailable.

    The failure this guards against is the tempting one: a serving layer that
    catches an upstream error and returns a default, a cached value, or the
    other tier's answer. Every one of those produces a plausible number that
    nobody can distinguish from a real one, and a loan decision made on it is
    indistinguishable from a correct decision until much later.

    ``responses`` are observations of the form
    ``{"status": int, "scored": bool}``.
    """
    fabricated = [r for r in responses if r.get("status") == 200 and r.get("scored")]
    return InvariantResult(
        name=NO_FABRICATED_PREDICTIONS,
        held=not fabricated,
        detail=(
            f"{len(responses)} request(s) while the model tier was unavailable; "
            f"{len(fabricated)} returned a score"
        ),
        evidence={
            "requests": len(responses),
            "scored_responses": len(fabricated),
            "status_codes": sorted({r.get("status") for r in responses}),
        },
    )


def production_alias_unchanged(before: str | None, after: str | None) -> InvariantResult:
    """The production alias points where it did before the failure.

    Checked around every scenario, not only the promotion ones. A failure that
    silently moved production would be the most damaging outcome available, and
    the cheapest to miss.
    """
    return InvariantResult(
        name=PRODUCTION_ALIAS_UNCHANGED,
        held=before == after,
        detail=f"production was v{before} before and v{after} after",
        evidence={"before": before, "after": after},
    )


def failed_candidate_not_production(
    *, promoted: bool, registered: bool, alias_before: str | None, alias_after: str | None
) -> InvariantResult:
    """A candidate the gates rejected is not registered and is not production."""
    held = (not promoted) and (not registered) and alias_before == alias_after
    return InvariantResult(
        name=FAILED_CANDIDATE_NOT_PRODUCTION,
        held=held,
        detail=(
            f"gates promoted={promoted}, registered={registered}, "
            f"production v{alias_before} -> v{alias_after}"
        ),
        evidence={
            "promoted": promoted,
            "registered": registered,
            "alias_before": alias_before,
            "alias_after": alias_after,
        },
    )


def canary_rollback_restores_incumbent(
    *, canary_before: int, canary_after: int, total_after: int
) -> InvariantResult:
    """After rollback, no request reaches the canary.

    Measured by counting the tier each response reports, not by reading the
    configuration back: the question is where requests actually went.
    """
    held = canary_after == 0 and total_after > 0
    return InvariantResult(
        name=CANARY_ROLLBACK_RESTORES_INCUMBENT,
        held=held,
        detail=(
            f"{canary_before} request(s) reached the canary before rollback, "
            f"{canary_after} of {total_after} after"
        ),
        evidence={
            "canary_before": canary_before,
            "canary_after": canary_after,
            "total_after": total_after,
        },
    )


def telemetry_failure_isolated(responses: list[dict[str, Any]]) -> InvariantResult:
    """Inference kept serving correct answers with telemetry gone.

    The inverse of :func:`no_fabricated_predictions`: here a 200 with a score is
    the success, because nothing about a metrics backend should reach a
    prediction.
    """
    served = [r for r in responses if r.get("status") == 200 and r.get("scored")]
    return InvariantResult(
        name=TELEMETRY_FAILURE_ISOLATED,
        held=len(served) == len(responses) and bool(responses),
        detail=(
            f"{len(served)} of {len(responses)} request(s) served normally "
            "while telemetry was unavailable"
        ),
        evidence={
            "requests": len(responses),
            "served": len(served),
            "status_codes": sorted({r.get("status") for r in responses}),
        },
    )


def recovery_returns_known_state(
    *,
    healthy: bool,
    model_identity_before: dict[str, Any],
    model_identity_after: dict[str, Any],
    detail: str = "",
) -> InvariantResult:
    """The system came back, serving the same model it served before.

    Both halves matter. Coming back unhealthy is an outage; coming back healthy
    but serving a *different* model is worse, because nothing looks wrong.
    """
    identical = model_identity_before == model_identity_after
    return InvariantResult(
        name=RECOVERY_RETURNS_KNOWN_STATE,
        held=healthy and identical,
        detail=(
            detail or f"healthy={healthy}, model identity {'unchanged' if identical else 'CHANGED'}"
        ),
        evidence={
            "healthy": healthy,
            "identity_before": model_identity_before,
            "identity_after": model_identity_after,
        },
    )


def dependency_failure_is_explicit(
    *, succeeded: bool, status: int | None, message: str
) -> InvariantResult:
    """An operation that genuinely needs a dead dependency failed, and said so.

    Silence is the failure mode being excluded: an operation that appears to
    succeed against a dependency that is not there has produced an answer from
    nowhere.
    """
    return InvariantResult(
        name=DEPENDENCY_FAILURE_IS_EXPLICIT,
        held=(not succeeded) and bool(message),
        detail=f"succeeded={succeeded}, status={status}, message={message[:160]!r}",
        evidence={"succeeded": succeeded, "status": status, "message": message[:400]},
    )


def inference_unaffected(responses: list[dict[str, Any]]) -> InvariantResult:
    """Every request was served normally while something else was broken.

    Same shape as :func:`telemetry_failure_isolated` but its own name, because a
    report that said "telemetry_failure_isolated" while the registry was the
    thing switched off would be describing an exercise nobody ran.
    """
    served = [r for r in responses if r.get("status") == 200 and r.get("scored")]
    return InvariantResult(
        name=INFERENCE_UNAFFECTED,
        held=len(served) == len(responses) and bool(responses),
        detail=f"{len(served)} of {len(responses)} request(s) served normally",
        evidence={
            "requests": len(responses),
            "served": len(served),
            "status_codes": sorted({r.get("status") for r in responses}),
        },
    )
