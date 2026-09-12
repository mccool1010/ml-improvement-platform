"""Promotion pipeline for M6.

Trains a candidate, pairs it against whatever is currently production, runs the
configured quality gates, and registers the candidate **only** if every mandatory
gate passed.

The order matters. Gates run before any registry call, so a rejected candidate
cannot reach the registry even transiently. A rejected candidate keeps its
ordinary MLflow run and its logged model artifact; what it does not get is a
registered version or the production alias.

No deployment happens here. Moving the alias is registry bookkeeping that lets
the next comparison know what the incumbent is. Serving and traffic shifting are
later milestones.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ml_platform.config import Config, load_config
from ml_platform.paths import ensure_dir
from ml_platform.pipelines.train_pipeline import RunRecord, run_training
from ml_platform.promotion.compare import Comparison, build_comparison
from ml_platform.promotion.gates import PromotionDecision, evaluate_gates
from ml_platform.promotion.registry import register_candidate

LOGGER = logging.getLogger(__name__)


def evaluate_candidate(config: Config, comparison: Comparison) -> PromotionDecision:
    """Run the gates. Pure decision, no side effects on the registry."""
    report = evaluate_gates(
        comparison.candidate_metrics,
        comparison.production_metrics,
        config.gate_config,
        context=comparison.context,
        split=comparison.decision_split,
        candidate_name=comparison.candidate.model_name,
        production_name=comparison.production.name,
        production_source=comparison.production.source,
    )
    return PromotionDecision(report=report, source_run_id=comparison.candidate.mlflow_run_id)


def run_promotion(
    environment: str = "production",
    *,
    config: Config | None = None,
    candidate: RunRecord | None = None,
    model_key: str = "candidate",
    param_overrides: dict[str, Any] | None = None,
    register: bool = True,
    nrows: int | None = None,
    alias: str | None = None,
) -> tuple[PromotionDecision, Comparison]:
    """Evaluate a candidate for promotion and register it if it passes.

    ``alias`` is passed straight through to
    :func:`ml_platform.promotion.registry.register_candidate`; ``None`` means the
    production alias, which is what every caller before M14 expects. M14 uses it
    to register a gated candidate under the canary alias, so traffic can reach it
    before it becomes production.
    """
    cfg = config or load_config(environment)

    record = candidate or run_training(
        config=cfg,
        model_key=model_key,
        save_model=False,
        nrows=nrows,
        param_overrides=param_overrides,
    )

    comparison = build_comparison(cfg, record, nrows=nrows)
    decision = evaluate_candidate(cfg, comparison)

    LOGGER.info(decision.report.summary())
    for gate in decision.report.gates:
        LOGGER.info("  %s", gate.describe())

    if decision.report.promote and register:
        name, version = register_candidate(cfg, record, decision.report, alias=alias)
        decision.registered = version is not None
        decision.registered_model = name
        decision.version = version
        if version is None:
            decision.notes.append("gates passed but registration did not complete")
    elif not decision.report.promote:
        decision.notes.append(
            "rejected: " + "; ".join(f"{g.name} ({g.reason})" for g in decision.report.failures)
        )
        LOGGER.warning("candidate rejected; production is unchanged")

    payload = {
        "decision": decision.to_dict(),
        "comparison": comparison.to_dict(),
    }
    destination = ensure_dir(cfg.benchmark_dir) / f"promotion-{record.context.run_id}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("wrote promotion report to %s", destination)
    return decision, comparison
