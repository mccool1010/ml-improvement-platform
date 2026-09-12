"""Read-only aggregation for the platform dashboard.

Every value served here is read from a system that already owns it: MLflow for
run lineage and registry state, the registry aliases for what production is, the
canary configuration for traffic allocation, Prometheus for request metrics, and
the failure harness's own scenario catalogue. Nothing in this module stores
state, decides anything, or serves a model.

Two rules shape all of it.

**Nothing is invented.** Each section reports ``available`` and, when false, the
reason. A dashboard showing "MLflow: unavailable" is telling the truth; one
showing a plausible last-known metric it made up is worse than one showing
nothing, because a reader cannot tell the difference. There is no fallback to a
cached or default value anywhere below.

**Server-side aggregation.** The browser never talks to MLflow or the Kubernetes
API. It would need credentials, network reach and knowledge of their internals,
and it would make the dashboard a second client of systems that already have
one. So the aggregation happens here and the browser sees plain JSON.

The endpoints are read-only. No control operation was needed: promotion,
retraining, canary decisions and failure injection all have CLIs that already
carry the safety checks, and putting a button in front of them would mean
duplicating those checks or bypassing them.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ml_platform.config import Config
from ml_platform.serving.canary import TIER_CANARY, TIER_PRODUCTION

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/platform", tags=["platform"])

#: Component health vocabulary. Three values, so a dashboard can colour them
#: without interpreting free text.
STATUS_HEALTHY = "healthy"
STATUS_UNAVAILABLE = "unavailable"
STATUS_DEGRADED = "degraded"

#: The lifecycle this platform implements, in order. Held here rather than in
#: the frontend so the stages and the milestone that built each cannot drift
#: apart from the code that runs them.
LIFECYCLE_STAGES: tuple[dict[str, str], ...] = (
    {"stage": "train", "component": "ml_platform.pipelines.train_pipeline", "milestone": "M1-M2"},
    {"stage": "experiment", "component": "MLflow tracking", "milestone": "M4"},
    {"stage": "optimize", "component": "Optuna study", "milestone": "M5"},
    {"stage": "evaluate", "component": "ml_platform.models.evaluate", "milestone": "M1"},
    {"stage": "compare", "component": "ml_platform.promotion.compare", "milestone": "M6"},
    {"stage": "quality gates", "component": "ml_platform.promotion.gates", "milestone": "M6"},
    {"stage": "promote / reject", "component": "MLflow Model Registry", "milestone": "M6"},
    {"stage": "deploy", "component": "Docker, Kubernetes, KServe", "milestone": "M8-M11"},
    {"stage": "monitor", "component": "Prometheus, Grafana, Jaeger", "milestone": "M12"},
    {"stage": "drift", "component": "ml_platform.monitoring.drift", "milestone": "M13"},
    {"stage": "retrain", "component": "ml_platform.pipelines.retrain_pipeline", "milestone": "M13"},
    {"stage": "canary", "component": "ml_platform.serving.canary", "milestone": "M14"},
    {"stage": "rollback", "component": "ml_platform.pipelines.canary_pipeline", "milestone": "M14"},
    {
        "stage": "failure / recovery",
        "component": "ml_platform.failure",
        "milestone": "M15",
    },
)

#: Why each quality gate blocks a promotion. Keyed by the gate names the
#: promotion code actually registers, and asserted against
#: ``gates.GATE_BUILDERS`` in the tests: a gate added or renamed there without
#: a description here is a test failure, not a silently stale dashboard.
GATE_RATIONALE: dict[str, str] = {
    "min_improvement": (
        "The candidate must beat the incumbent by a stated margin. Ties and noise "
        "are not improvements, and churning production for them costs more than it wins."
    ),
    "minimum_metric": (
        "An absolute floor, so a weak incumbent cannot lower the bar for its successor."
    ),
    "roc_auc_regression": ("Ranking quality must not regress while the headline metric improves."),
    "calibration": (
        "Brier skill must hold. A well-ranked model with wrong probabilities is "
        "unusable at a fixed decision threshold."
    ),
    "recall_regression": (
        "Recall at the review capacity must not fall: that is the number the "
        "business outcome is actually made of."
    ),
    "latency": ("A model too slow to serve is not an improvement, however good its scores."),
    "reproducibility": (
        "The candidate must be reproducible from its recorded provenance, on a "
        "clean revision with a recorded lockfile."
    ),
}

#: What falsifies each failure invariant. Keyed by the names
#: ``ml_platform.failure.invariants`` defines, and asserted against that module
#: in the tests, so the dashboard cannot describe an invariant the harness has
#: renamed or dropped.
INVARIANT_FALSIFIED_BY: dict[str, str] = {
    "no_fabricated_predictions": "any 200-with-a-score while the model tier is down",
    "production_alias_unchanged": "the alias moving during a failure",
    "failed_candidate_not_production": "a rejected candidate being registered or aliased",
    "canary_rollback_restores_incumbent": ("any request still reaching the canary after rollback"),
    "telemetry_failure_isolated": ("an inference failing because a metrics backend is down"),
    "inference_unaffected": "the same, for a non-telemetry dependency",
    "recovery_returns_known_state": (
        "coming back unhealthy, or coming back serving a different model"
    ),
    "dependency_failure_is_explicit": (
        "an operation succeeding against a dependency that is not there"
    ),
}

#: Validation metrics worth showing for a production model, in display order.
#: Names match the run records exactly; nothing is renamed for presentation.
HEADLINE_METRICS: tuple[str, ...] = (
    "average_precision",
    "roc_auc",
    "recall_at_capacity",
    "precision_at_capacity",
    "brier_score",
    "brier_skill_score",
)


# --- response models --------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    status: str
    detail: str | None = None
    #: What this component is authoritative for, so a reader knows what its
    #: being down actually costs.
    owns: str | None = None


class PlatformHealth(BaseModel):
    status: str
    components: list[ComponentHealth]
    #: True only when every component a prediction needs is healthy.
    can_serve: bool


class Section(BaseModel):
    """Base for anything that may genuinely have nothing to report."""

    available: bool
    detail: str | None = None


class ProductionModel(Section):
    name: str | None = None
    version: str | None = None
    alias: str | None = None
    feature_set: str | None = None
    mlflow_run_id: str | None = None
    platform_run_id: str | None = None
    decision_split: str | None = None
    gates_passed: int | None = None
    gates_total: int | None = None
    git_revision: str | None = None
    dataset_sha256: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    served_by: str | None = None
    decision_threshold: float | None = None
    threshold_source: str | None = None


class VersionSummary(BaseModel):
    version: str
    aliases: list[str] = Field(default_factory=list)
    is_production: bool = False
    mlflow_run_id: str | None = None
    gates_passed: int | None = None
    gates_total: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str | None = None


class PromotionHistory(Section):
    registered_model: str | None = None
    production_version: str | None = None
    versions: list[VersionSummary] = Field(default_factory=list)
    #: Runs that were trained and never registered. The honest denominator:
    #: without it a registry looks like everything ever tried was promoted.
    unregistered_candidates: int | None = None


class FeatureDriftSummary(BaseModel):
    feature: str
    psi: float
    drifted: bool


class DriftState(Section):
    decision: str | None = None
    drift_detected: bool | None = None
    n_drifted: int | None = None
    n_features: int | None = None
    max_psi: float | None = None
    threshold_psi: float | None = None
    min_drifted_features: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    scenario: str | None = None
    checked_at: str | None = None
    top_features: list[FeatureDriftSummary] = Field(default_factory=list)
    retraining: dict[str, Any] = Field(default_factory=dict)
    #: Stated on the payload, not only in the docs: there is no live realised
    #: performance signal, and the dashboard must not imply one.
    performance_signal_available: bool = False
    performance_note: str = (
        "Realised performance needs matured labels, which take 60 months on this "
        "dataset. Drift measures inputs only and cannot say a model got worse."
    )


class CanaryState(Section):
    traffic_percent: float = 0.0
    active: bool = False
    incumbent_version: str | None = None
    candidate_version: str | None = None
    last_decision: str | None = None
    last_reason: str | None = None
    decided_at: str | None = None
    signals: dict[str, float] = Field(default_factory=dict)
    rollback_signals: list[str] = Field(default_factory=list)
    accuracy_used_as_signal: bool = False
    signal_note: str = (
        "Rollback uses operational signals only: error rate, latency, upstream "
        "failures and serving health. Accuracy is excluded because labels lag by "
        "years, so a canary waiting for it would never conclude."
    )


class FailureScenario(BaseModel):
    scenario: str
    mode: str
    invariants: list[str] = Field(default_factory=list)
    passed: bool | None = None
    observed: str | None = None


class FailureInvariant(BaseModel):
    name: str
    falsified_by: str


class FailureState(Section):
    scenarios: list[FailureScenario] = Field(default_factory=list)
    invariants: list[FailureInvariant] = Field(default_factory=list)
    evidence_available: bool = False
    last_run_id: str | None = None
    n_passed: int | None = None


class ObservabilitySummary(Section):
    request_rate: float | None = None
    error_ratio: float | None = None
    latency_p50_seconds: float | None = None
    latency_p95_seconds: float | None = None
    model_tier_p95_seconds: float | None = None
    applications_scored: float | None = None
    #: Grafana and Jaeger remain the observability systems; these are pointers,
    #: not a reimplementation.
    grafana_url: str | None = None
    jaeger_url: str | None = None
    prometheus_url: str | None = None


class QualityGate(BaseModel):
    name: str
    rationale: str
    #: Whether this installation has the gate configured for promotion runs.
    configured: bool


class PlatformEvent(BaseModel):
    at: str
    kind: str
    summary: str
    outcome: str | None = None


class LifecycleStage(BaseModel):
    stage: str
    component: str
    milestone: str
    state: str | None = None


# --- data access ------------------------------------------------------------


def _config(request: Request) -> Config:
    config = getattr(request.app.state, "config", None)
    if isinstance(config, Config):
        return config
    from ml_platform.config import load_config  # pragma: no cover - misbuilt app

    return load_config("production")


class _Mlflow:
    """A narrow MLflow reader that never raises into a response."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.error: str | None = None
        self._client: Any = None

    def client(self) -> Any:
        if self._client is None:
            try:
                self._bound_retries()
                import mlflow

                mlflow.set_tracking_uri(self.config.tracking_uri)
                self._client = mlflow.MlflowClient()
                # Cheap call that actually touches the backend, so an
                # unreachable server is discovered here rather than halfway
                # through building a response.
                self._client.search_experiments(max_results=1)
            except Exception as exc:
                self.error = f"MLflow is unavailable: {type(exc).__name__}"
                LOGGER.info(self.error)
                self._client = None
        return self._client

    @staticmethod
    def _bound_retries() -> None:
        """Fail fast when the registry is gone.

        The MLflow client retries a connection failure with exponential
        backoff, which is right for a training run and wrong for a dashboard
        request: a browser would sit on a spinner for a minute before being
        told MLflow is down. The same bounds the M12 ConfigMap sets on the
        pods, applied here so the aggregation answers promptly wherever it
        runs. Only set if the operator has not chosen their own values.
        """
        import os

        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

    def runs(self, *, run_type: str | None = None, limit: int = 25) -> list[Any]:
        client = self.client()
        if client is None:
            return []
        try:
            experiment = client.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                return []
            filter_string = f"tags.run_type = '{run_type}'" if run_type else ""
            return list(
                client.search_runs(
                    [experiment.experiment_id],
                    filter_string=filter_string,
                    order_by=["attributes.start_time DESC"],
                    max_results=limit,
                )
            )
        except Exception as exc:
            self.error = f"MLflow query failed: {type(exc).__name__}"
            return []

    def training_runs(self, limit: int = 200) -> list[Any]:
        """Runs that produced a model, however they were produced.

        Training runs carry ``model_name`` and no ``run_type``; retraining
        candidates carry both. Filtering on ``run_type = 'training'`` matched
        neither, which made the unregistered-candidate count zero and hid every
        training event -- the opposite of the honest denominator it was meant
        to be. Presence of ``model_name`` is what actually marks a run as having
        produced a model, so that is what is asked.
        """
        return [run for run in self.runs(limit=limit) if run.data.tags.get("model_name")]

    def versions(self) -> list[Any]:
        client = self.client()
        if client is None:
            return []
        try:
            return list(client.search_model_versions(f"name='{self.config.registered_model_name}'"))
        except Exception as exc:
            self.error = f"registry query failed: {type(exc).__name__}"
            return []


def _prometheus_query(base_url: str, expression: str, timeout: float = 5.0) -> float | None:
    """One instant query, or ``None``. Never raises."""
    try:
        import httpx

        response = httpx.get(
            f"{base_url.rstrip('/')}/api/v1/query", params={"query": expression}, timeout=timeout
        )
        response.raise_for_status()
        result = response.json()["data"]["result"]
    except Exception:
        return None
    if not result:
        return None
    value = float(result[0]["value"][1])
    # NaN is what a quantile over an empty window returns. Reporting it as 0
    # would read as "instant responses" rather than "no traffic".
    return None if value != value else value


def _timestamp(run: Any) -> str | None:
    try:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat()
    except Exception:  # pragma: no cover
        return None


def _metrics_from_tags(tags: dict[str, str], split: str) -> dict[str, float]:
    """Validation metrics the registry recorded on a version."""
    out: dict[str, float] = {}
    for metric in HEADLINE_METRICS:
        raw = tags.get(f"{split}_{metric}")
        if raw is None:
            continue
        try:
            out[metric] = float(raw)
        except ValueError:
            continue
    return out


# --- endpoints --------------------------------------------------------------


@router.get("/health", response_model=PlatformHealth)
def platform_health(request: Request) -> PlatformHealth:
    """Aggregate health of every component the platform depends on.

    Reuses the M7 model service for the serving answer rather than probing
    again, so this endpoint and ``/ready`` can never disagree.
    """
    config = _config(request)
    service = getattr(request.app.state, "model_service", None)

    components: list[ComponentHealth] = [
        ComponentHealth(
            name="API",
            status=STATUS_HEALTHY,
            detail="this process is serving",
            owns="request validation, thresholds, version reporting",
        )
    ]

    can_serve = False
    if service is None:  # pragma: no cover - misbuilt app
        components.append(
            ComponentHealth(name="Model serving", status=STATUS_UNAVAILABLE, detail="no service")
        )
    else:
        ready, detail = service.readiness()
        can_serve = ready
        tier = "KServe" if (service.loaded and service.model.predictor) else "in-process"
        components.append(
            ComponentHealth(
                name="Model serving",
                status=STATUS_HEALTHY if ready else STATUS_UNAVAILABLE,
                detail=detail or f"serving via {tier}",
                owns="model artifact, feature building, scoring",
            )
        )

    mlflow_reader = _Mlflow(config)
    mlflow_up = mlflow_reader.client() is not None
    components.append(
        ComponentHealth(
            name="MLflow",
            status=STATUS_HEALTHY if mlflow_up else STATUS_UNAVAILABLE,
            detail=mlflow_reader.error or "registry and run lineage reachable",
            owns="experiment lineage, model registry, promotion history",
        )
    )

    prometheus_url = config.prometheus_url
    prometheus_up = _prometheus_query(prometheus_url, "up") is not None if prometheus_url else False
    components.append(
        ComponentHealth(
            name="Prometheus",
            status=STATUS_HEALTHY if prometheus_up else STATUS_UNAVAILABLE,
            detail=None if prometheus_up else "not reachable from the API",
            owns="request metrics, canary signals",
        )
    )

    # Overall status reflects what a caller can actually do. MLflow or
    # Prometheus being down is degraded, not unhealthy: predictions still work,
    # which is the whole point of keeping them off the request path.
    if not can_serve:
        overall = STATUS_UNAVAILABLE
    elif not (mlflow_up and prometheus_up):
        overall = STATUS_DEGRADED
    else:
        overall = STATUS_HEALTHY

    return PlatformHealth(status=overall, components=components, can_serve=can_serve)


@router.get("/model", response_model=ProductionModel)
def production_model(request: Request) -> ProductionModel:
    """What production is, and the evidence that put it there."""
    config = _config(request)
    service = getattr(request.app.state, "model_service", None)
    reader = _Mlflow(config)
    client = reader.client()

    if client is None:
        return ProductionModel(available=False, detail=reader.error)

    try:
        version = client.get_model_version_by_alias(
            config.registered_model_name, config.production_alias
        )
    except Exception:
        return ProductionModel(
            available=False,
            detail=(
                f"no model carries the {config.production_alias!r} alias on "
                f"{config.registered_model_name!r}"
            ),
        )

    tags = dict(version.tags or {})
    model = ProductionModel(
        available=True,
        name=config.registered_model_name,
        version=str(version.version),
        alias=config.production_alias,
        feature_set=tags.get("feature_set"),
        mlflow_run_id=str(version.run_id),
        platform_run_id=tags.get("platform_run_id"),
        decision_split=tags.get("decision_split"),
        gates_passed=_as_int(tags.get("gates_passed")),
        gates_total=_as_int(tags.get("gates_total")),
        git_revision=tags.get("git_revision"),
        dataset_sha256=tags.get("dataset_sha256"),
        metrics=_metrics_from_tags(tags, tags.get("decision_split") or config.decision_split),
    )

    # Serving-side facts come from the live service, not the registry: what is
    # loaded right now can differ from what the alias says, and that difference
    # is exactly what a reader needs to see.
    if service is not None and service.loaded:
        loaded = service.model
        model.served_by = loaded.served_by
        model.decision_threshold = loaded.decision_threshold
        model.threshold_source = loaded.threshold_source
    return model


@router.get("/lifecycle", response_model=list[LifecycleStage])
def lifecycle(request: Request) -> list[LifecycleStage]:
    """The lifecycle stages, and which of them have real state to show."""
    config = _config(request)
    reader = _Mlflow(config)
    have_runs = bool(reader.runs(limit=1))
    have_drift = bool(reader.runs(run_type="drift_check", limit=1))
    have_canary = bool(reader.runs(run_type="canary_decision", limit=1))
    have_retrain = bool(reader.runs(run_type="retraining_candidate", limit=1))
    service = getattr(request.app.state, "model_service", None)
    serving = bool(service and service.loaded)

    observed = {
        "train": have_runs,
        "experiment": have_runs,
        "evaluate": have_runs,
        "compare": have_runs,
        "quality gates": have_runs,
        "promote / reject": have_runs,
        "deploy": serving,
        "monitor": config.prometheus_url is not None,
        "drift": have_drift,
        "retrain": have_retrain,
        "canary": have_canary,
        "rollback": have_canary,
    }
    return [
        LifecycleStage(
            **stage,
            state="observed" if observed.get(stage["stage"], False) else "implemented",
        )
        for stage in LIFECYCLE_STAGES
    ]


@router.get("/promotions", response_model=PromotionHistory)
def promotions(request: Request) -> PromotionHistory:
    """Registered versions, their gate results, and how many never made it."""
    config = _config(request)
    reader = _Mlflow(config)
    versions = reader.versions()
    if not versions:
        return PromotionHistory(available=False, detail=reader.error or "no registered versions")

    production_version: str | None = None
    summaries: list[VersionSummary] = []
    for version in sorted(versions, key=lambda v: int(v.version), reverse=True):
        tags = dict(version.tags or {})
        aliases = [str(a) for a in (getattr(version, "aliases", None) or [])]
        is_production = config.production_alias in aliases
        if is_production:
            production_version = str(version.version)
        summaries.append(
            VersionSummary(
                version=str(version.version),
                aliases=aliases,
                is_production=is_production,
                mlflow_run_id=str(version.run_id),
                gates_passed=_as_int(tags.get("gates_passed")),
                gates_total=_as_int(tags.get("gates_total")),
                metrics=_metrics_from_tags(tags, tags.get("decision_split") or "validation"),
                created_at=_iso_millis(getattr(version, "creation_timestamp", None)),
            )
        )

    if production_version is None:
        # The alias may be set without being reflected on the version object,
        # depending on the backend, so ask the registry directly.
        client = reader.client()
        try:
            resolved = client.get_model_version_by_alias(
                config.registered_model_name, config.production_alias
            )
            production_version = str(resolved.version)
            for summary in summaries:
                summary.is_production = summary.version == production_version
        except Exception:
            production_version = None

    trained = reader.training_runs()
    registered_runs = {s.mlflow_run_id for s in summaries}
    return PromotionHistory(
        available=True,
        registered_model=config.registered_model_name,
        production_version=production_version,
        versions=summaries,
        unregistered_candidates=sum(1 for run in trained if run.info.run_id not in registered_runs),
    )


@router.get("/drift", response_model=DriftState)
def drift(request: Request) -> DriftState:
    """The most recent drift check, and what retraining did about it."""
    config = _config(request)
    reader = _Mlflow(config)
    checks = reader.runs(run_type="drift_check", limit=1)
    if not checks:
        return DriftState(
            available=False,
            detail=reader.error or "no drift check has been run; use `python -m ml_platform drift`",
        )

    run = checks[0]
    tags = dict(run.data.tags)
    metrics = dict(run.data.metrics)
    params = dict(run.data.params)

    top = sorted(
        (
            FeatureDriftSummary(
                feature=key.removeprefix("psi_"),
                psi=value,
                drifted=value >= float(params.get("threshold_psi", 0.1) or 0.1),
            )
            for key, value in metrics.items()
            if key.startswith("psi_")
        ),
        key=lambda f: -f.psi,
    )[:8]

    state = DriftState(
        available=True,
        decision=tags.get("drift_decision"),
        drift_detected=tags.get("drift_decision") == "retrain",
        n_drifted=_as_int(metrics.get("n_drifted_features")),
        n_features=len(top) and _as_int(len([k for k in metrics if k.startswith("psi_")])),
        max_psi=metrics.get("max_psi"),
        threshold_psi=_as_float(params.get("threshold_psi")),
        min_drifted_features=_as_int(params.get("min_drifted_features")),
        window_start=tags.get("drift_window_start"),
        window_end=tags.get("drift_window_end"),
        scenario=tags.get("drift_scenario"),
        checked_at=_timestamp(run),
        top_features=top,
    )

    event_id = tags.get("drift_event_id")
    if event_id:
        caused = [
            r
            for r in reader.runs(run_type="retraining_candidate", limit=20)
            if r.data.tags.get("drift_event_id") == event_id
        ]
        if caused:
            candidate = caused[0]
            state.retraining = {
                "ran": True,
                "candidate_run_id": candidate.info.run_id,
                "validation_average_precision": candidate.data.metrics.get(
                    "validation_average_precision"
                ),
                "window_rows": candidate.data.tags.get("retrain_window_rows"),
                "promoted": candidate.info.run_id in {v.run_id for v in reader.versions()},
            }
        else:
            state.retraining = {"ran": False}
    return state


@router.get("/canary", response_model=CanaryState)
def canary(request: Request) -> CanaryState:
    """Traffic allocation now, plus the last decision the evaluator reached."""
    config = _config(request)
    router_state = getattr(request.app.state, "canary_router", None)
    reader = _Mlflow(config)

    state = CanaryState(
        available=True,
        rollback_signals=[
            "candidate health",
            "HTTP error rate (absolute and vs incumbent)",
            "upstream failure rate",
            "latency p95 (absolute and vs incumbent)",
        ],
    )
    if router_state is not None:
        live = router_state.state
        state.traffic_percent = router_state.traffic_percent
        state.active = live.active
        state.candidate_version = live.candidate_version
        state.incumbent_version = live.incumbent_version

    decisions = reader.runs(run_type="canary_decision", limit=1)
    if decisions:
        run = decisions[0]
        tags = dict(run.data.tags)
        state.last_decision = tags.get("canary_decision")
        state.last_reason = tags.get("canary_reason")
        state.decided_at = _timestamp(run)
        state.candidate_version = state.candidate_version or tags.get("candidate_version")
        state.incumbent_version = state.incumbent_version or tags.get("incumbent_version")
        state.signals = {
            key: value
            for key, value in run.data.metrics.items()
            if key.startswith((f"{TIER_CANARY}_", f"{TIER_PRODUCTION}_"))
        }
    elif reader.error:
        state.detail = reader.error
    return state


@router.get("/failure", response_model=FailureState)
def failure(request: Request) -> FailureState:
    """The failure scenarios this platform defines, and any recorded evidence.

    The catalogue comes from the harness itself, so it cannot list a scenario
    that does not exist. Evidence is only reported when a report is actually
    readable; the harness writes reports where it runs, which is not inside a
    pod.
    """
    del request
    from ml_platform.failure import invariants as inv
    from ml_platform.failure.runner import LIVE_SCENARIOS, OFFLINE_SCENARIOS

    scenarios = [
        FailureScenario(scenario=name, mode="live-kubernetes") for name in sorted(LIVE_SCENARIOS)
    ] + [
        FailureScenario(scenario=name, mode="controlled-double")
        for name in sorted(OFFLINE_SCENARIOS)
    ]
    # The invariant names come from the harness module, not from a list held
    # anywhere nearer the frontend.
    names = [
        value
        for key, value in vars(inv).items()
        if key.isupper() and isinstance(value, str) and not key.startswith("_")
    ]
    return FailureState(
        available=True,
        scenarios=scenarios,
        invariants=[
            FailureInvariant(name=name, falsified_by=INVARIANT_FALSIFIED_BY[name])
            for name in names
            if name in INVARIANT_FALSIFIED_BY
        ],
        evidence_available=False,
        detail=(
            "Scenario evidence is written where the harness runs, not inside the cluster. "
            "Run `python -m ml_platform failure` to produce it; recorded results are in "
            "docs/failure.md."
        ),
    )


@router.get("/observability", response_model=ObservabilitySummary)
def observability(request: Request) -> ObservabilitySummary:
    """A small summary from Prometheus, plus pointers to the real tools.

    Six numbers, not a dashboard. Grafana and Jaeger remain the observability
    systems and this does not try to replace either.
    """
    config = _config(request)
    prometheus_url = config.prometheus_url
    summary = ObservabilitySummary(
        available=False,
        prometheus_url=prometheus_url,
        grafana_url=config.grafana_url,
        jaeger_url=config.jaeger_url,
    )
    if not prometheus_url:
        summary.detail = "no Prometheus endpoint is configured"
        return summary

    service = 'service="inference-api"'
    queries = {
        "request_rate": f"sum(rate(ml_platform_http_requests_total{{{service}}}[5m]))",
        "error_ratio": (
            f"sum(rate(ml_platform_http_request_errors_total{{{service}}}[5m])) / "
            f"clamp_min(sum(rate(ml_platform_http_requests_total{{{service}}}[5m])), 0.001)"
        ),
        "latency_p50_seconds": (
            "histogram_quantile(0.50, sum by (le) (rate("
            f'ml_platform_http_request_duration_seconds_bucket{{{service},route="/predict"}}[5m])))'
        ),
        "latency_p95_seconds": (
            "histogram_quantile(0.95, sum by (le) (rate("
            f'ml_platform_http_request_duration_seconds_bucket{{{service},route="/predict"}}[5m])))'
        ),
        "model_tier_p95_seconds": (
            "histogram_quantile(0.95, sum by (le) (rate("
            "ml_platform_model_tier_duration_seconds_bucket[5m])))"
        ),
        "applications_scored": "sum(ml_platform_applications_scored_total)",
    }
    values = {name: _prometheus_query(prometheus_url, q) for name, q in queries.items()}
    if all(value is None for value in values.values()):
        summary.detail = "Prometheus is not reachable from the API"
        return summary

    summary.available = True
    for name, value in values.items():
        setattr(summary, name, value)
    return summary


@router.get("/gates", response_model=list[QualityGate])
def quality_gates(request: Request) -> list[QualityGate]:
    """The promotion gates, in the order the promotion code runs them.

    The names come from ``gates.GATE_BUILDERS`` rather than from a list in the
    frontend, so a gate cannot be added, renamed or removed without this
    catalogue following it.
    """
    from ml_platform.promotion.gates import GATE_BUILDERS

    configured = set(_config(request).gate_config or {})
    return [
        QualityGate(
            name=name,
            rationale=GATE_RATIONALE.get(name, ""),
            configured=name in configured,
        )
        for name in GATE_BUILDERS
    ]


@router.get("/events", response_model=list[PlatformEvent])
def events(request: Request, limit: int = 12) -> list[PlatformEvent]:
    """Recent platform activity, assembled from MLflow runs.

    Derived rather than stored: there is no event log to keep in sync, and a
    second one would be a second source of truth about what happened.
    """
    config = _config(request)
    reader = _Mlflow(config)
    collected: list[PlatformEvent] = []

    registered_runs = {v.run_id: str(v.version) for v in reader.versions()}

    for run in reader.runs(limit=60):
        tags = dict(run.data.tags)
        # Training runs carry no run_type; `model_name` is what marks them.
        kind = tags.get("run_type", "")
        at = _timestamp(run)
        if at is None:
            continue

        if kind == "drift_check":
            decision = tags.get("drift_decision", "unknown")
            drifted = _as_int(run.data.metrics.get("n_drifted_features"))
            collected.append(
                PlatformEvent(
                    at=at,
                    kind="drift",
                    summary=f"Drift check on {tags.get('drift_window_start')}"
                    f"..{tags.get('drift_window_end')}: {drifted} feature(s) over threshold",
                    outcome=decision,
                )
            )
        elif kind == "retraining_candidate":
            promoted = run.info.run_id in registered_runs
            collected.append(
                PlatformEvent(
                    at=at,
                    kind="retraining",
                    summary="Retrained candidate offered to the quality gates",
                    outcome="promoted" if promoted else "rejected",
                )
            )
        elif kind == "canary_decision":
            collected.append(
                PlatformEvent(
                    at=at,
                    kind="canary",
                    summary=tags.get("canary_reason") or "Canary evaluated",
                    outcome=tags.get("canary_decision"),
                )
            )
        elif tags.get("model_name"):
            version = registered_runs.get(run.info.run_id)
            name = tags["model_name"]
            collected.append(
                PlatformEvent(
                    at=at,
                    kind="training",
                    summary=f"Trained {name}",
                    outcome=f"registered as v{version}" if version else "not registered",
                )
            )

    return collected[:limit]


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_millis(value: Any) -> str | None:
    try:
        from datetime import UTC, datetime

        return datetime.fromtimestamp(float(value) / 1000, tz=UTC).isoformat()
    except (TypeError, ValueError):
        return None
