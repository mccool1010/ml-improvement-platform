"""Running the scenarios and collecting their evidence."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ml_platform.config import Config
from ml_platform.failure.cluster import ClusterControl
from ml_platform.failure.report import FailureReport, ScenarioEvidence, new_run_id
from ml_platform.failure.scenarios import (
    scenario_bad_candidate,
    scenario_canary_failure,
    scenario_mlflow_failure,
    scenario_model_serving_failure,
    scenario_restart_recovery,
    scenario_telemetry_failure,
)

LOGGER = logging.getLogger(__name__)

#: Scenarios that break a real component. They need a cluster and a reachable API.
LIVE_SCENARIOS: dict[str, Callable[..., ScenarioEvidence]] = {
    "model_serving_failure": scenario_model_serving_failure,
    "mlflow_dependency_failure": scenario_mlflow_failure,
    "canary_failure": scenario_canary_failure,
    "telemetry_failure": scenario_telemetry_failure,
    "restart_recovery": scenario_restart_recovery,
}

#: Scenarios that need neither, and say so in their report.
OFFLINE_SCENARIOS: dict[str, Callable[..., ScenarioEvidence]] = {
    "bad_candidate_promotion": scenario_bad_candidate,
}

ALL_SCENARIOS = [*LIVE_SCENARIOS, *OFFLINE_SCENARIOS]


def run(
    config: Config,
    *,
    base_url: str,
    names: list[str] | None = None,
    namespace: str = "ml-platform",
    live: bool = True,
    offline_config: Config | None = None,
    tracking_uri: str | None = None,
) -> FailureReport:
    """Run the named scenarios, restoring everything as it goes.

    ``live`` gates anything that touches the cluster, so the default for a test
    or a laptop is to run only what is safe there.

    ``tracking_uri`` points the registry checks at the cluster's MLflow rather
    than whatever the local configuration names. Without it a live run asks the
    laptop's SQLite store whether the cluster's registry is up, which it will
    cheerfully answer. It is set for the duration of the run and restored
    afterwards: the environment variable outranks configuration, so leaving it
    behind would silently redirect every later registry read in the process.
    """
    from ml_platform.config import override_tracking_uri

    previous: str | None = None
    if tracking_uri:
        previous = override_tracking_uri(tracking_uri)
        LOGGER.info("registry checks will use %s", tracking_uri)
    chosen = names or ALL_SCENARIOS
    report = FailureReport(run_id=new_run_id())
    cluster = ClusterControl(namespace=namespace)

    if live and not cluster.available():
        LOGGER.warning("no cluster is reachable; live scenarios will be skipped")
        live = False

    try:
        for name in chosen:
            if name in LIVE_SCENARIOS:
                if not live:
                    LOGGER.info("skipping live scenario %s", name)
                    continue
                LOGGER.info("running live scenario %s", name)
                evidence = LIVE_SCENARIOS[name](cluster, base_url, config)
            elif name in OFFLINE_SCENARIOS:
                LOGGER.info("running scenario %s", name)
                evidence = OFFLINE_SCENARIOS[name](offline_config or config)
            else:
                raise ValueError(f"unknown scenario {name!r}; known: {ALL_SCENARIOS}")

            LOGGER.info(evidence.summary())
            report.scenarios.append(evidence)
    finally:
        # The same discipline every injector in this harness follows: whatever
        # was changed to run a scenario is put back, including process state.
        if tracking_uri:
            override_tracking_uri(previous)

    return report


def describe(report: FailureReport) -> str:
    """A readable rendering of a report, used by the CLI."""
    lines: list[str] = [report.summary(), ""]
    for evidence in report.scenarios:
        lines.append(f"--- {evidence.scenario} [{evidence.mode}] ---")
        lines.append(f"  injected   {evidence.failure_injected}")
        lines.append(f"  expected   {evidence.expected_behaviour}")
        lines.append(f"  observed   {evidence.observed_behaviour}")
        for item in evidence.blast_radius:
            lines.append(f"  blast      {item}")
        for item in evidence.unaffected:
            lines.append(f"  unaffected {item}")
        lines.append(f"  recovery   {evidence.recovery_action}")
        lines.append(f"             {evidence.recovery_result}")
        for invariant in evidence.invariants:
            lines.append(f"  {invariant.describe()}")
        lines.append(f"  {evidence.summary()}")
        lines.append("")
    return "\n".join(lines)
