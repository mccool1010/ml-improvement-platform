"""Running a canary and deciding what happens to it.

    python -m ml_platform canary --action start|evaluate|rollback|complete

The sequence, and the reason each step exists:

1. **start** -- a candidate that has already passed the M6 gates is registered
   under the *canary* alias, not the production one. Gates judged its offline
   quality; nothing yet knows how it behaves under real requests.
2. traffic -- the application tier routes a configured share to it.
3. **evaluate** -- operational signals over an observation window produce a
   decision: promote, rollback, or hold.
4. **complete** -- on promote, and only then, the production alias moves.
   **rollback** -- traffic returns to the incumbent and the alias is untouched.

The safety property is that the production alias moves in exactly one place in
this module, guarded by a decision object that had to say ``promote``. There is
no path from "a candidate exists" to "the candidate is production" that skips
either the gates or the canary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_platform.config import Config, load_config
from ml_platform.monitoring.canary import (
    DECISION_PROMOTE,
    DECISION_ROLLBACK,
    CanaryDecision,
    SignalSource,
    evaluate_canary,
)
from ml_platform.paths import ensure_dir
from ml_platform.promotion.registry import resolve_production
from ml_platform.serving.canary import TIER_CANARY, TIER_PRODUCTION, CanaryRouter
from ml_platform.serving.traffic import TrafficController

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.pipelines.train_pipeline import RunRecord

LOGGER = logging.getLogger(__name__)


class CanaryStateError(RuntimeError):
    """Raised when an action does not make sense for the current state."""


def new_canary_event_id() -> str:
    return "canary-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class CanaryOutcome:
    """A completed canary, and what it did to the registry."""

    canary_event_id: str
    decision: CanaryDecision
    alias_moved: bool
    production_version: str | None
    report_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.decision.should_promote and self.alias_moved:
            return f"canary passed; production is now v{self.production_version}"
        if self.decision.should_rollback:
            return (
                f"canary rolled back; production remains v{self.production_version}: "
                f"{self.decision.reason}"
            )
        return f"canary held; production remains v{self.production_version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "canary_event_id": self.canary_event_id,
            "alias_moved": self.alias_moved,
            "production_version": self.production_version,
            "notes": self.notes,
            **self.decision.to_dict(),
        }


def start_canary(
    config: Config,
    record: RunRecord,
    report: Any,
    *,
    traffic_percent: float,
    router: CanaryRouter | None = None,
    candidate_url: str | None = None,
) -> tuple[str | None, CanaryRouter]:
    """Register a gated candidate under the canary alias and open traffic to it.

    ``report`` is the M6 gate report. It is passed straight to
    ``register_candidate``, which refuses a candidate whose gates did not pass --
    so a canary cannot be a way around the gates, only an additional hurdle
    after them.
    """
    from ml_platform.promotion.registry import register_candidate

    incumbent = resolve_production(config)
    name, version = register_candidate(config, record, report, alias=config.canary_alias)
    if version is None:
        raise CanaryStateError("the candidate could not be registered; no canary was started")

    LOGGER.info(
        "registered %s v%s under the %r alias; production remains v%s",
        name,
        version,
        config.canary_alias,
        incumbent.version if incumbent else "none",
    )

    active = router or CanaryRouter()
    active.start(
        traffic_percent=traffic_percent,
        candidate_url=candidate_url or str(config.canary_predictor_url or "http://canary"),
        candidate_version=version,
        incumbent_version=incumbent.version if incumbent else None,
    )
    _publish_traffic(active.traffic_percent)
    return version, active


def evaluate(
    config: Config,
    signals: SignalSource,
    *,
    canary_event_id: str,
    router: CanaryRouter,
    window_seconds: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> CanaryDecision:
    """Measure both tiers and reach a verdict. No side effects."""
    window = window_seconds or config.canary_observation_seconds
    return evaluate_canary(
        canary=signals.signals_for(TIER_CANARY, window),
        production=signals.signals_for(TIER_PRODUCTION, window),
        thresholds=config.canary_thresholds,
        canary_event_id=canary_event_id,
        traffic_percent=router.state.traffic_percent,
        observation_window_seconds=window,
        candidate_version=router.state.candidate_version,
        incumbent_version=router.state.incumbent_version,
        metadata={"signal_source": type(signals).__name__, **(metadata or {})},
    )


def rollback(
    config: Config,
    router: CanaryRouter,
    decision: CanaryDecision,
    *,
    controller: TrafficController | None = None,
) -> CanaryOutcome:
    """Return every request to the incumbent, leaving the registry alone.

    Two things happen, and the order matters. Traffic stops first, because that
    is what actually protects callers. The registry is then *not* touched: the
    production alias still points where it did before the canary started, so
    there is nothing to restore and no window in which production is ambiguous.

    ``controller`` is what makes the stop reach the processes serving real
    traffic. Without one this moves only the router held here, which is right
    for a rollback triggered inside the API and useless for one triggered from a
    command line -- the process would exit having changed nothing. The CLI
    passes a :class:`~ml_platform.serving.traffic.KubernetesTrafficController`.
    """
    router.stop(decision.reason)
    _publish_traffic(0.0)

    applied = "this process only"
    if controller is not None:
        controller.set_traffic(0.0)
        applied = controller.describe()
        LOGGER.info("canary traffic set to zero via %s", applied)

    incumbent = resolve_production(config)
    outcome = CanaryOutcome(
        canary_event_id=decision.canary_event_id,
        decision=decision,
        alias_moved=False,
        production_version=incumbent.version if incumbent else None,
        notes=[
            f"traffic returned to the incumbent ({applied})",
            "the production alias was never moved",
        ],
    )
    LOGGER.warning(outcome.summary())
    return outcome


def complete(
    config: Config,
    router: CanaryRouter,
    decision: CanaryDecision,
    *,
    controller: TrafficController | None = None,
) -> CanaryOutcome:
    """Make the candidate production. The only place the alias moves.

    Three preconditions, each guarding a different way this could go wrong.

    The decision must say ``promote``, so a failed or undecided canary cannot be
    completed.

    A canary must still be **running**. Without this, an old ``CanaryDecision``
    object is a loaded gun: replaying one after the canary ended would move the
    alias again, and if production had since advanced to a newer version it
    would move it *backwards* to the one in the stale report.

    The running canary must be for **this candidate**. A decision measured
    against version 2 says nothing about a version 3 that started afterwards, so
    completing one with the other's evidence is refused.
    """
    if decision.decision != DECISION_PROMOTE:
        raise CanaryStateError(
            f"refusing to complete a canary whose decision was {decision.decision!r}: "
            f"{decision.reason}"
        )
    if not router.state.active:
        raise CanaryStateError(
            "refusing to complete: no canary is running. A decision from a canary that has "
            "already ended cannot be replayed, because production may have moved on since."
        )
    version = decision.candidate_version or router.state.candidate_version
    if version is None:
        raise CanaryStateError("no candidate version to promote")
    if router.state.candidate_version is not None and version != router.state.candidate_version:
        raise CanaryStateError(
            f"refusing to complete: the decision is for v{version} but the running canary is "
            f"v{router.state.candidate_version}. Evidence gathered for one candidate does not "
            "justify promoting another."
        )

    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            config.registered_model_name, config.production_alias, version
        )
        # The canary alias is cleared: leaving it behind would make the next
        # canary's "is one running" question answer yes forever.
        try:
            client.delete_registered_model_alias(config.registered_model_name, config.canary_alias)
        except Exception:  # pragma: no cover - absent alias is fine
            LOGGER.debug("no %r alias to clear", config.canary_alias)
    except Exception as exc:
        raise CanaryStateError(f"could not move the production alias: {exc}") from exc

    # All traffic now goes to what is, from here, simply production.
    router.stop("canary completed; the candidate is now production")
    _publish_traffic(0.0)
    if controller is not None:
        controller.set_traffic(0.0)

    LOGGER.info("production alias moved to v%s", version)
    return CanaryOutcome(
        canary_event_id=decision.canary_event_id,
        decision=decision,
        alias_moved=True,
        production_version=version,
        notes=[f"production alias moved to v{version}", "canary alias cleared"],
    )


def decide(
    config: Config,
    router: CanaryRouter,
    decision: CanaryDecision,
    *,
    controller: TrafficController | None = None,
) -> CanaryOutcome:
    """Act on a decision: complete it, roll it back, or leave it running."""
    if decision.decision == DECISION_PROMOTE:
        return complete(config, router, decision, controller=controller)
    if decision.decision == DECISION_ROLLBACK:
        return rollback(config, router, decision, controller=controller)

    incumbent = resolve_production(config)
    return CanaryOutcome(
        canary_event_id=decision.canary_event_id,
        decision=decision,
        alias_moved=False,
        production_version=incumbent.version if incumbent else None,
        notes=["not enough evidence yet; the canary is still running"],
    )


def write_report(config: Config, outcome: CanaryOutcome) -> str:
    """Persist the decision report and return its path."""
    destination = ensure_dir(config.benchmark_dir) / f"{outcome.canary_event_id}.json"
    Path(destination).write_text(json.dumps(outcome.to_dict(), indent=2, default=str), "utf-8")
    outcome.report_path = str(destination)
    LOGGER.info("wrote canary report to %s", destination)
    return str(destination)


def log_to_mlflow(config: Config, outcome: CanaryOutcome) -> str | None:
    """Record the canary decision against the candidate's lineage.

    Tagged with the same ``canary_event_id`` the report carries, so a registered
    version can be traced to the traffic that justified it.
    """
    if not config.tracking_enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        decision = outcome.decision
        with mlflow.start_run(run_name=outcome.canary_event_id) as run:
            mlflow.set_tags(
                {
                    "run_type": "canary_decision",
                    "canary_event_id": outcome.canary_event_id,
                    "canary_decision": decision.decision,
                    "canary_reason": decision.reason[:480],
                    "candidate_version": str(decision.candidate_version),
                    "incumbent_version": str(decision.incumbent_version),
                    "alias_moved": str(outcome.alias_moved),
                }
            )
            mlflow.log_params(
                {
                    "traffic_percent": decision.traffic_percent,
                    "observation_window_seconds": decision.observation_window_seconds,
                    **{f"threshold_{k}": v for k, v in decision.thresholds.items()},
                }
            )
            for tier, signals in (
                (TIER_CANARY, decision.canary),
                (TIER_PRODUCTION, decision.production),
            ):
                mlflow.log_metrics(
                    {
                        f"{tier}_requests": signals.requests,
                        f"{tier}_errors": signals.errors,
                        f"{tier}_error_rate": signals.error_rate,
                        f"{tier}_failure_rate": signals.failure_rate,
                        f"{tier}_latency_p95": signals.latency_p95_seconds,
                    }
                )
            if outcome.report_path:
                mlflow.log_artifact(outcome.report_path)
            return str(run.info.run_id)
    except Exception:
        LOGGER.warning("could not log the canary decision to MLflow", exc_info=True)
        return None


def _publish_traffic(percent: float) -> None:
    from ml_platform.observability.metrics import set_canary_traffic

    set_canary_traffic(percent)


def load_config_for(environment: str) -> Config:
    return load_config(environment)
