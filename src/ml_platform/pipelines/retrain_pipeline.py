"""Retraining, triggered by drift and judged by the existing gates.

    python -m ml_platform retrain [--scenario lending_shift] [--force]

The shape of this pipeline is the whole point of the milestone:

    drift check -> (maybe) retrain on the new window -> M6 gates -> registry

Retraining produces a *candidate*, never a production model. The candidate goes
through :func:`ml_platform.pipelines.promote_pipeline.run_promotion` -- the same
function `python -m ml_platform promote` calls -- so the same seven gates decide,
on the same validation split, against whatever currently holds the alias. A
retrained model that is worse is rejected and production is untouched. Nothing
here can move the alias; only `register_candidate` can, and only when the gates
passed.

**On the labels this uses.** Retraining trains on rows from the production
window, which have labels. They have them because this is a historical register
whose 60-month horizon has fully elapsed for those approvals -- matured labels,
not invented ones. In a live system those labels would not exist yet, and the
window would have to wait five years before it could be trained on. The maturity
cutoff is enforced below rather than assumed, and the gap is documented in
docs/drift.md, because pretending otherwise is the failure this project exists
to argue against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from ml_platform.config import Config, load_config
from ml_platform.data.splitting import make_splits
from ml_platform.monitoring.windows import build_current_window
from ml_platform.paths import ensure_dir
from ml_platform.pipelines.drift_pipeline import DriftEvent, run_drift_check
from ml_platform.pipelines.promote_pipeline import run_promotion
from ml_platform.pipelines.train_pipeline import load_prepared_dataset, run_training

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.promotion.compare import Comparison
    from ml_platform.promotion.gates import PromotionDecision

LOGGER = logging.getLogger(__name__)


class MaturityError(RuntimeError):
    """Raised when a window is not old enough to have usable labels."""


@dataclass
class RetrainOutcome:
    """What a retraining attempt did, and what was decided about it."""

    drift_event: DriftEvent
    triggered: bool
    reason: str
    candidate_run_id: str | None = None
    decision: PromotionDecision | None = None
    comparison: Comparison | None = None
    report_path: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        return bool(self.decision and self.decision.report.promote)

    @property
    def registered(self) -> bool:
        return bool(self.decision and self.decision.registered)

    def summary(self) -> str:
        if not self.triggered:
            return f"no retraining: {self.reason}"
        if self.decision is None:  # pragma: no cover - defensive
            return "retrained, but no decision was reached"
        verdict = "PROMOTED" if self.promoted else "REJECTED"
        return f"retrained and {verdict}: {self.decision.report.summary()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_event_id": self.drift_event.event_id,
            "drift_decision": self.drift_event.decision,
            "triggered": self.triggered,
            "reason": self.reason,
            "candidate_run_id": self.candidate_run_id,
            "promoted": self.promoted,
            "registered": self.registered,
            "decision": self.decision.to_dict() if self.decision else None,
            "window": self.drift_event.window.describe(),
            "notes": self.notes,
        }


def assert_labels_matured(config: Config, window_end: Any) -> None:
    """Refuse to train on rows whose outcome could not yet be known.

    The check is the milestone's honesty guarantee. Without it, "retrain on the
    current window" quietly becomes "train on outcomes from the future", which
    would produce a model that looks excellent and could never exist.
    """
    cutoff = config.label_maturity_end
    if window_end > cutoff:
        raise MaturityError(
            f"the window ends {window_end}, past the label maturity cutoff {cutoff}. "
            f"Those loans have not had their {config.horizon_months}-month horizon elapse "
            "before the observation cutoff, so their outcomes are not known and must not "
            "be trained on."
        )


def run_retraining(
    environment: str = "production",
    *,
    config: Config | None = None,
    scenario: str | None = None,
    model_key: str = "candidate",
    force: bool = False,
    register: bool = True,
    param_overrides: dict[str, Any] | None = None,
    nrows: int | None = None,
    drift_event: DriftEvent | None = None,
) -> RetrainOutcome:
    """Check for drift and, if it is found, retrain and offer the result up.

    ``force`` retrains regardless of the drift verdict, for a scheduled refresh
    or a deliberate experiment. It changes what triggers retraining; it changes
    nothing about how the result is judged.
    """
    cfg = config or load_config(environment)
    event = drift_event or run_drift_check(config=cfg, scenario=scenario, nrows=nrows)

    if not event.should_retrain and not force:
        LOGGER.info("no drift; production is left alone")
        return RetrainOutcome(
            drift_event=event,
            triggered=False,
            reason=f"drift check returned {event.decision!r}",
        )

    reason = (
        "forced by request"
        if force and not event.should_retrain
        else f"{event.report.n_drifted} feature(s) drifted past psi={event.report.threshold_psi}"
    )
    LOGGER.info("retraining: %s", reason)

    # The window is rebuilt rather than carried on the event, so this function is
    # usable with a drift event loaded from disk.
    prepared, _ = load_prepared_dataset(cfg, nrows=nrows)
    window = build_current_window(
        prepared,
        cfg,
        start=event.window.start,
        end=event.window.end,
        scenario=event.window.scenario,
        seed=event.window.seed,
    )
    # Asserted on the rows that will actually be trained on, not on the date
    # that was asked for. Those differ whenever a window runs past the data.
    latest = pd.to_datetime(window.frame[cfg.split_date_column]).max().date()
    assert_labels_matured(cfg, latest)

    # Guard against silently training on evaluation rows. The windows are
    # configured not to overlap, but a future edit could change that and the
    # failure would show up only as an inexplicably good candidate.
    splits = make_splits(prepared, cfg)
    _assert_disjoint_from_evaluation(window.frame, splits, cfg)

    record = run_training(
        config=cfg,
        model_key=model_key,
        save_model=False,
        nrows=nrows,
        param_overrides=param_overrides,
        extra_training_frame=window.frame,
        training_note=(
            f"drift window {window.start}..{window.end} "
            f"scenario={window.scenario} ({event.event_id})"
        ),
    )

    # Lineage, on the candidate's own MLflow run: the drift event that caused it,
    # and the exact rows it was retrained on.
    _tag_candidate(cfg, record, event, window)

    decision, comparison = run_promotion(
        config=cfg,
        candidate=record,
        model_key=model_key,
        register=register,
        nrows=nrows,
    )

    outcome = RetrainOutcome(
        drift_event=event,
        triggered=True,
        reason=reason,
        candidate_run_id=record.context.run_id,
        decision=decision,
        comparison=comparison,
    )
    if not outcome.promoted:
        outcome.notes.append("gates rejected the retrained candidate; production is unchanged")

    destination = ensure_dir(cfg.benchmark_dir) / f"retrain-{event.event_id}.json"
    destination.write_text(
        json.dumps(
            {"generated_at": datetime.now(UTC).isoformat(), **outcome.to_dict()},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    outcome.report_path = str(destination)
    LOGGER.info("wrote retraining report to %s", destination)
    LOGGER.info(outcome.summary())
    return outcome


def _assert_disjoint_from_evaluation(
    window: pd.DataFrame, splits: dict[str, Any], config: Config
) -> None:
    """The retraining window may not contain evaluation rows."""
    date_column = config.split_date_column
    window_dates = pd.to_datetime(window[date_column])
    for name in ("validation", "test"):
        if name not in splits:
            continue
        bounds = splits[name].window
        overlap = (
            (window_dates >= pd.Timestamp(bounds.start))
            & (window_dates <= pd.Timestamp(bounds.end))
        ).sum()
        if overlap:
            raise ValueError(
                f"the retraining window overlaps the {name} split by {overlap} row(s). "
                "Training on evaluation rows would make every gate meaningless."
            )


def _tag_candidate(config: Config, record: Any, event: DriftEvent, window: Any) -> None:
    """Join the candidate's MLflow run to the drift event that caused it."""
    if not (config.tracking_enabled and record.mlflow_run_id):
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        client = mlflow.MlflowClient()
        tags = {
            "run_type": "retraining_candidate",
            "drift_event_id": event.event_id,
            "drift_decision": event.decision,
            "drift_mlflow_run_id": str(event.mlflow_run_id),
            "retrain_window_start": str(window.start),
            "retrain_window_end": str(window.end),
            "retrain_window_scenario": window.scenario,
            "retrain_window_fingerprint": window.fingerprint(),
            "retrain_window_rows": str(window.n_rows),
        }
        for key, value in tags.items():
            client.set_tag(record.mlflow_run_id, key, value)
    except Exception:
        LOGGER.warning("could not tag the candidate with its drift lineage", exc_info=True)
