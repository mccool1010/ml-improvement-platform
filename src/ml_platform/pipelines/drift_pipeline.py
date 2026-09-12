"""Run a drift check and record it as a traceable event.

    python -m ml_platform drift [--scenario lending_shift]

Produces a JSON report and a `drift_event_id` that every downstream artefact
carries, so a promotion months later can be traced back to the check that
prompted it. The event is logged to MLflow as an ordinary run, which is what
makes the chain queryable alongside the training runs it caused.

The decision this returns is `retrain` or `no_action`, and never anything that
changes production. Drift observes inputs; the M6 gates decide what is served.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_platform.config import Config, load_config
from ml_platform.data.splitting import make_splits
from ml_platform.monitoring import drift as drift_module
from ml_platform.monitoring.windows import CurrentWindow, build_current_window
from ml_platform.paths import ensure_dir, relative_to_root
from ml_platform.pipelines.train_pipeline import load_prepared_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """One drift check: what was compared, what was found, what it implies."""

    event_id: str
    report: drift_module.DriftReport
    window: CurrentWindow
    report_path: str
    mlflow_run_id: str | None = None

    @property
    def decision(self) -> str:
        return self.report.decision

    @property
    def should_retrain(self) -> bool:
        return self.report.drift_detected

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_event_id": self.event_id,
            "decision": self.decision,
            "window": self.window.describe(),
            "report_path": self.report_path,
            "mlflow_run_id": self.mlflow_run_id,
            "report": self.report.to_dict(),
        }


def _new_event_id() -> str:
    return "drift-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log_to_mlflow(config: Config, event: DriftEvent) -> str | None:
    """Record the check as an MLflow run. Never fatal.

    Tracking is where the lineage becomes queryable, but a tracking server that
    is down must not stop a drift check from producing its answer -- the JSON
    report is the deliverable, exactly as it is for training runs.
    """
    if not config.tracking_enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name=event.event_id) as run:
            report = event.report
            mlflow.set_tags(
                {
                    "run_type": "drift_check",
                    "drift_event_id": event.event_id,
                    "drift_decision": report.decision,
                    "drift_scenario": event.window.scenario,
                    "drift_window_start": str(event.window.start),
                    "drift_window_end": str(event.window.end),
                    "drift_window_fingerprint": event.window.fingerprint(),
                    "drift_reference_split": config.drift_reference_split,
                }
            )
            mlflow.log_params(
                {
                    "threshold_psi": report.threshold_psi,
                    "min_drifted_features": report.min_drifted_features,
                    "feature_set": report.feature_set,
                }
            )
            mlflow.log_metrics(
                {
                    "n_drifted_features": report.n_drifted,
                    "share_drifted_features": report.share_drifted,
                    "max_psi": report.max_psi,
                    "reference_rows": report.reference_rows,
                    "current_rows": report.current_rows,
                }
            )
            # Per-feature PSI, so a dashboard or a later query can see which
            # features moved without opening the JSON.
            for feature in report.features:
                mlflow.log_metric(f"psi_{feature.feature}", feature.psi)
            mlflow.log_artifact(event.report_path)
            return str(run.info.run_id)
    except Exception:
        LOGGER.warning("could not log the drift event to MLflow", exc_info=True)
        return None


def run_drift_check(
    environment: str = "production",
    *,
    config: Config | None = None,
    scenario: str | None = None,
    nrows: int | None = None,
) -> DriftEvent:
    """Compare the reference split against the configured production window."""
    cfg = config or load_config(environment)

    prepared, checksum = load_prepared_dataset(cfg, nrows=nrows)
    splits = make_splits(prepared, cfg)
    reference = splits[cfg.drift_reference_split].frame

    window = build_current_window(prepared, cfg, scenario=scenario)
    event_id = _new_event_id()

    report = drift_module.compare_with_config(
        cfg,
        reference,
        window.frame,
        metadata={
            "drift_event_id": event_id,
            "reference_split": cfg.drift_reference_split,
            "dataset_sha256": checksum,
            "window": window.describe(),
        },
    )

    destination = ensure_dir(cfg.report_dir) / f"{event_id}.json"
    payload = {
        "drift_event_id": event_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "window": window.describe(),
        **report.to_dict(),
    }
    Path(destination).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote drift report to %s", destination)

    event = DriftEvent(
        event_id=event_id,
        report=report,
        window=window,
        report_path=str(destination),
    )
    event.mlflow_run_id = _log_to_mlflow(cfg, event)
    LOGGER.info("%s -> %s", event_id, report.summary())
    return event


def describe_event(event: DriftEvent) -> str:
    """A short human summary, used by the CLI."""
    lines = [
        f"drift event   {event.event_id}",
        f"window        {event.window.start}..{event.window.end} "
        f"scenario={event.window.scenario} rows={event.window.n_rows}",
        f"fingerprint   {event.window.fingerprint()}",
        f"report        {relative_to_root(Path(event.report_path))}",
        "",
        event.report.summary(),
    ]
    for feature in sorted(event.report.features, key=lambda f: -f.psi)[:8]:
        lines.append(f"  {feature.describe()}")
    if event.mlflow_run_id:
        lines.append(f"\nmlflow run    {event.mlflow_run_id}")
    return "\n".join(lines)
