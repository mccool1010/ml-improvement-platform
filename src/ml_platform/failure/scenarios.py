"""The failure scenarios, and what each one proves.

Each scenario follows the same shape: record the state that must not change,
break one real thing, observe, put it back, and check the invariant. The
``finally`` that restores is not optional -- a harness that can leave the cluster
broken is a worse liability than the failures it tests for.

Five of the six break a real component with ``kubectl``. One, the bad-candidate
promotion, uses a controlled double: proving that the gates reject a bad model
does not require destroying the real registry, and doing so would produce worse
evidence, not better. That scenario runs the genuine promotion pipeline, the
genuine seven gates and the genuine registration code against an isolated store,
and its report says ``controlled-double`` so nobody has to guess.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from ml_platform.config import Config
from ml_platform.failure import invariants as inv
from ml_platform.failure.cluster import ClusterControl
from ml_platform.failure.report import MODE_DOUBLE, MODE_LIVE, ScenarioEvidence

LOGGER = logging.getLogger(__name__)

API_DEPLOYMENT = "inference-api"
PREDICTOR_DEPLOYMENT = "sba-loan-default-predictor"
CANARY_PREDICTOR_DEPLOYMENT = "sba-loan-default-canary-predictor"
MLFLOW_DEPLOYMENT = "mlflow"
PROMETHEUS_DEPLOYMENT = "prometheus"
JAEGER_DEPLOYMENT = "jaeger"
API_CONFIGMAP = "inference-api-config"

#: One valid application, used for every probe. Deliberately the same record
#: throughout so a changed score means a changed model, not a changed input.
APPLICATION: dict[str, Any] = {
    "term_months": 84,
    "employees": 12,
    "jobs_created": 3,
    "jobs_retained": 8,
    "gross_approved": 250000.0,
    "sba_approved": 187500.0,
    "disbursed": 250000.0,
    "state": "CA",
    "bank_state": "CA",
    "revolving_line_of_credit": "N",
    "low_doc": "N",
    "urban_rural": 1,
    "new_business": 1,
    "naics": "722410",
    "franchise_code": 0,
    "approval_date": "2005-06-15",
    "disbursement_date": "2005-07-20",
}


# --- probing ----------------------------------------------------------------


def probe_predict(base_url: str, routing_key: str, timeout: float = 30.0) -> dict[str, Any]:
    """One prediction attempt, recorded as an observation rather than raised.

    ``scored`` is the field the invariants turn on: whether a *number* came
    back, not merely whether the call returned.
    """
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/predict",
        data=json.dumps({"applications": [APPLICATION]}).encode(),
        headers={"content-type": "application/json", "x-ml-routing-key": routing_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
            predictions = payload.get("predictions") or []
            return {
                "status": response.status,
                "scored": bool(predictions),
                "probability": predictions[0]["default_probability"] if predictions else None,
                "tier": payload.get("model", {}).get("serving_tier"),
                "version": payload.get("model", {}).get("version"),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        return {"status": exc.code, "scored": False, "detail": body}
    except Exception as exc:
        return {"status": None, "scored": False, "detail": f"{type(exc).__name__}: {exc}"}


def probe_get(base_url: str, path: str, timeout: float = 20.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
            body = response.read().decode(errors="replace")
            return {"status": response.status, "body": body[:600]}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode(errors="replace")[:600]}
    except Exception as exc:
        return {"status": None, "body": f"{type(exc).__name__}: {exc}"}


def _model_identity(base_url: str) -> dict[str, Any]:
    """Which model the service says it is serving, for before/after comparison."""
    observed = probe_get(base_url, "/ready")
    try:
        model = json.loads(observed["body"]).get("model") or {}
    except Exception:
        return {}
    return {
        "name": model.get("name"),
        "version": model.get("version"),
        "alias": model.get("alias"),
        "served_by": model.get("served_by"),
    }


def _identity_from_pod(cluster: ClusterControl) -> dict[str, Any]:
    """Model identity read through the Service from inside the cluster."""
    observed = cluster.ready_from_pod(API_DEPLOYMENT, "http://inference-api/ready")
    try:
        model = json.loads(observed.get("body") or "{}").get("model") or {}
    except Exception:
        return {}
    return {
        "name": model.get("name"),
        "version": model.get("version"),
        "alias": model.get("alias"),
        "served_by": model.get("served_by"),
    }


def _production_version(config: Config) -> str | None:
    """The registry's current production version, read through M6's own resolver.

    Which registry this reaches is decided by ``config.tracking_uri``, and that
    matters more than it looks. Run from a laptop against the default
    configuration it reads the *local* SQLite store, so a scenario that switched
    off the cluster's MLflow and then called this would be asking a completely
    different registry whether it was alive -- and getting a confident yes. The
    live runner therefore sets MLFLOW_TRACKING_URI at the cluster's server
    before any scenario starts. The first run of the MLflow scenario made
    exactly this mistake, and the invariant caught it.
    """
    try:
        from ml_platform.promotion.registry import resolve_production

        production = resolve_production(config)
        return production.version if production else None
    except Exception:  # pragma: no cover - a dead registry is itself an observation
        return None


# --- scenario 1: the model tier disappears ----------------------------------


def scenario_model_serving_failure(
    cluster: ClusterControl, base_url: str, config: Config
) -> ScenarioEvidence:
    """Scale the KServe predictor to zero and ask for a prediction."""
    evidence = ScenarioEvidence(
        scenario="model_serving_failure",
        description="The KServe model tier is unavailable.",
        mode=MODE_LIVE,
        failure_injected=f"kubectl scale deployment/{PREDICTOR_DEPLOYMENT} --replicas=0",
        expected_behaviour=(
            "Requests fail explicitly with 5xx and no score. Readiness turns false so the "
            "Service withholds traffic. The production alias does not move."
        ),
    )
    alias_before = _production_version(config)
    scaled = cluster.scale(PREDICTOR_DEPLOYMENT, 0)
    try:
        cluster.wait_gone(PREDICTOR_DEPLOYMENT)
        time.sleep(5)

        responses = [probe_predict(base_url, f"fail-{i}") for i in range(8)]
        ready = probe_get(base_url, "/ready")
        health = probe_get(base_url, "/health")

        evidence.observations = {
            "predict_responses": responses,
            "ready_status": ready["status"],
            "health_status": health["status"],
        }
        evidence.observed_behaviour = (
            f"/predict returned {sorted({r['status'] for r in responses})} with no scores; "
            f"/ready {ready['status']}; /health {health['status']}"
        )
        evidence.blast_radius = [
            "predictions unavailable",
            "readiness false, so the Service removes the pods from its endpoints",
        ]
        evidence.unaffected = [
            "liveness still 200: the API process is healthy and is not restarted",
            "the registry is untouched",
        ]

        evidence.add(inv.no_fabricated_predictions(responses))
        evidence.add(
            inv.dependency_failure_is_explicit(
                succeeded=any(r["scored"] for r in responses),
                status=responses[0]["status"] if responses else None,
                message=str(responses[0].get("detail", "")) if responses else "",
            )
        )
    finally:
        evidence.recovery_action = (
            f"scale deployment/{PREDICTOR_DEPLOYMENT} back to {scaled.replicas}"
        )
        cluster.restore(scaled)
        recovered = cluster.wait_serving(PREDICTOR_DEPLOYMENT)
        time.sleep(8)
        after = probe_predict(base_url, "recovered")
        evidence.recovered = recovered and after["scored"]
        evidence.recovery_result = (
            f"predictor ready={recovered}; a prediction returned {after.get('probability')}"
        )
        evidence.add(inv.production_alias_unchanged(alias_before, _production_version(config)))
        evidence.finish()
    return evidence


# --- scenario 2: the registry disappears ------------------------------------


def scenario_mlflow_failure(
    cluster: ClusterControl, base_url: str, config: Config
) -> ScenarioEvidence:
    """Scale MLflow to zero. Serving should not notice; registry work should."""
    evidence = ScenarioEvidence(
        scenario="mlflow_dependency_failure",
        description="The MLflow tracking server and registry are unavailable.",
        mode=MODE_LIVE,
        failure_injected=f"kubectl scale deployment/{MLFLOW_DEPLOYMENT} --replicas=0",
        expected_behaviour=(
            "Inference is unaffected: the model was resolved at startup and scoring goes to "
            "KServe, so MLflow is not on the request path. Operations that genuinely need "
            "the registry fail with a clear error rather than silently."
        ),
    )
    identity_before = _model_identity(base_url)
    scaled = cluster.scale(MLFLOW_DEPLOYMENT, 0)
    try:
        cluster.wait_gone(MLFLOW_DEPLOYMENT)
        time.sleep(5)

        responses = [probe_predict(base_url, f"mlflow-{i}") for i in range(8)]
        # The registry read is the operation that genuinely needs MLflow.
        resolved = _production_version(config)

        evidence.observations = {
            "predict_responses": responses,
            "registry_lookup_returned": resolved,
            "identity_during": _model_identity(base_url),
        }
        evidence.observed_behaviour = (
            f"/predict returned {sorted({r['status'] for r in responses})} with "
            f"{sum(1 for r in responses if r['scored'])} score(s); "
            f"a registry lookup returned {resolved!r}"
        )
        evidence.blast_radius = [
            "registry reads and writes unavailable: promotion, canary completion and "
            "drift logging cannot run",
            "a NEW api pod starting now would fail to resolve a model and stay unready",
        ]
        evidence.unaffected = [
            "inference: the model is already loaded and scoring goes to KServe",
            "the production alias, which nothing can move while the registry is down",
        ]

        evidence.add(inv.inference_unaffected(responses))
        evidence.add(
            inv.dependency_failure_is_explicit(
                succeeded=resolved is not None,
                status=None,
                message=(
                    "resolve_production returned None and logged that no alias could be read"
                    if resolved is None
                    else "the registry answered while it was supposed to be down"
                ),
            )
        )
    finally:
        evidence.recovery_action = f"scale deployment/{MLFLOW_DEPLOYMENT} back to {scaled.replicas}"
        cluster.restore(scaled)
        recovered = cluster.wait_serving(MLFLOW_DEPLOYMENT)
        time.sleep(10)
        evidence.recovered = recovered
        evidence.recovery_result = (
            f"mlflow ready={recovered}; registry lookup returns v{_production_version(config)}"
        )
        evidence.add(
            inv.recovery_returns_known_state(
                healthy=recovered,
                model_identity_before=identity_before,
                model_identity_after=_model_identity(base_url),
            )
        )
        evidence.finish()
    return evidence


# --- scenario 4: the canary tier fails --------------------------------------


def scenario_canary_failure(
    cluster: ClusterControl,
    base_url: str,
    config: Config,
    *,
    traffic_percent: float = 40.0,
) -> ScenarioEvidence:
    """Allocate traffic to a canary, break it, and roll back through M14.

    ``base_url`` is accepted for a uniform scenario signature but deliberately
    unused: every probe here goes through a pod, because changing the allocation
    restarts the API and an external port-forward does not survive that.
    """
    del base_url
    from ml_platform.serving.traffic import KubernetesTrafficController

    evidence = ScenarioEvidence(
        scenario="canary_failure",
        description="A canary receiving live traffic becomes unable to serve.",
        mode=MODE_LIVE,
        failure_injected=(
            f"allocate {traffic_percent}% of traffic to the canary, then "
            f"kubectl scale deployment/{CANARY_PREDICTOR_DEPLOYMENT} --replicas=0"
        ),
        expected_behaviour=(
            "Canary-routed requests fail rather than falling back to the incumbent. A "
            "rollback returns every request to the incumbent. The production alias never "
            "moves, so there is nothing to restore in the registry."
        ),
    )
    alias_before = _production_version(config)
    controller = KubernetesTrafficController(namespace=cluster.namespace)
    scaled = cluster.scale(CANARY_PREDICTOR_DEPLOYMENT, 0)
    try:
        cluster.wait_gone(CANARY_PREDICTOR_DEPLOYMENT)
        controller.set_traffic(traffic_percent)
        time.sleep(5)

        # Probed from inside the cluster: changing the allocation restarts the
        # API, and an external port-forward does not survive that.
        cluster.wait_ready(API_DEPLOYMENT)
        payload = json.dumps({"applications": [APPLICATION]})
        during = [
            cluster.probe_from_pod(
                API_DEPLOYMENT, "http://inference-api/predict", payload, f"canary-{i}"
            )
            for i in range(20)
        ]
        canary_before = sum(1 for r in during if r.get("tier") == "canary" or r["status"] == 503)
        failed = [r for r in during if not r["scored"]]

        # The rollback: the same controller the CLI uses.
        controller.set_traffic(0.0)
        cluster.wait_ready(API_DEPLOYMENT)
        time.sleep(5)
        after = [
            cluster.probe_from_pod(
                API_DEPLOYMENT, "http://inference-api/predict", payload, f"canary-{i}"
            )
            for i in range(20)
        ]
        canary_after = sum(1 for r in after if r.get("tier") == "canary")

        evidence.observations = {
            "during_status_codes": sorted({r["status"] for r in during}),
            "during_failures": len(failed),
            "after_status_codes": sorted({r["status"] for r in after}),
            "canary_requests_after": canary_after,
        }
        evidence.observed_behaviour = (
            f"with the canary dead, {len(failed)} of {len(during)} requests failed; "
            f"after rollback {canary_after} of {len(after)} reached the canary"
        )
        evidence.blast_radius = [
            f"approximately {traffic_percent}% of requests failed while the canary held traffic",
            "no fallback to the incumbent, by design: a masked failure would read as success",
        ]
        evidence.unaffected = [
            "the incumbent tier kept serving its share throughout",
            "the production alias",
        ]

        evidence.add(inv.no_fabricated_predictions(failed))
        evidence.add(
            inv.canary_rollback_restores_incumbent(
                canary_before=canary_before, canary_after=canary_after, total_after=len(after)
            )
        )
    finally:
        evidence.recovery_action = "set canary traffic to 0 and scale the canary predictor back up"
        try:
            controller.set_traffic(0.0)
        except Exception:  # pragma: no cover - best effort
            LOGGER.warning("could not reset canary traffic", exc_info=True)
        cluster.restore(scaled)
        recovered = cluster.wait_serving(CANARY_PREDICTOR_DEPLOYMENT)
        time.sleep(5)
        cluster.wait_ready(API_DEPLOYMENT)
        healthy = cluster.probe_from_pod(
            API_DEPLOYMENT,
            "http://inference-api/predict",
            json.dumps({"applications": [APPLICATION]}),
            "canary-recovered",
        )
        evidence.recovered = recovered and healthy["scored"]
        evidence.recovery_result = (
            f"canary predictor ready={recovered}; traffic at "
            f"{cluster.configmap_value(API_CONFIGMAP, 'ML_PLATFORM_CANARY_TRAFFIC_PERCENT')}%; "
            f"a prediction returned {healthy.get('probability')}"
        )
        evidence.add(inv.production_alias_unchanged(alias_before, _production_version(config)))
        evidence.finish()
    return evidence


# --- scenario 5: telemetry disappears ---------------------------------------


def scenario_telemetry_failure(
    cluster: ClusterControl, base_url: str, config: Config
) -> ScenarioEvidence:
    """Scale Prometheus and Jaeger to zero and keep serving."""
    evidence = ScenarioEvidence(
        scenario="telemetry_failure",
        description="Prometheus and Jaeger are both unavailable.",
        mode=MODE_LIVE,
        failure_injected=(
            f"kubectl scale deployment/{PROMETHEUS_DEPLOYMENT} deployment/{JAEGER_DEPLOYMENT} "
            "--replicas=0"
        ),
        expected_behaviour=(
            "Inference is entirely unaffected. Span export fails on a background thread and "
            "is logged; /metrics keeps being served even though nobody is scraping it."
        ),
    )
    alias_before = _production_version(config)
    scaled = [cluster.scale(PROMETHEUS_DEPLOYMENT, 0), cluster.scale(JAEGER_DEPLOYMENT, 0)]
    try:
        for deployment in (PROMETHEUS_DEPLOYMENT, JAEGER_DEPLOYMENT):
            cluster.wait_gone(deployment)
        time.sleep(5)

        responses = [probe_predict(base_url, f"telemetry-{i}") for i in range(12)]
        metrics = probe_get(base_url, "/metrics")
        ready = probe_get(base_url, "/ready")

        evidence.observations = {
            "predict_responses": responses,
            "metrics_status": metrics["status"],
            "ready_status": ready["status"],
            "distinct_probabilities": sorted(
                {r["probability"] for r in responses if r["probability"] is not None}
            ),
        }
        evidence.observed_behaviour = (
            f"{sum(1 for r in responses if r['scored'])} of {len(responses)} requests served "
            f"normally; /metrics {metrics['status']}; /ready {ready['status']}"
        )
        evidence.blast_radius = [
            "no metrics are collected and no traces are stored for the duration",
            "canary evaluation would be unable to read signals from Prometheus",
        ]
        evidence.unaffected = ["inference", "readiness", "the registry"]

        evidence.add(inv.telemetry_failure_isolated(responses))
    finally:
        evidence.recovery_action = "scale prometheus and jaeger back up"
        for state in scaled:
            cluster.restore(state)
        recovered = all(cluster.wait_serving(d) for d in (PROMETHEUS_DEPLOYMENT, JAEGER_DEPLOYMENT))
        evidence.recovered = recovered
        evidence.recovery_result = f"prometheus and jaeger ready={recovered}"
        evidence.add(inv.production_alias_unchanged(alias_before, _production_version(config)))
        evidence.finish()
    return evidence


# --- scenario 6: restart ----------------------------------------------------


def scenario_restart_recovery(
    cluster: ClusterControl, base_url: str, config: Config
) -> ScenarioEvidence:
    """Restart the API and the model tier; the system must come back the same."""
    evidence = ScenarioEvidence(
        scenario="restart_recovery",
        description="The API and the model tier are restarted.",
        mode=MODE_LIVE,
        failure_injected=(
            f"kubectl rollout restart deployment/{PREDICTOR_DEPLOYMENT} deployment/{API_DEPLOYMENT}"
        ),
        expected_behaviour=(
            "Both come back healthy and serve the same model version through the same tier. "
            "Neither the production alias nor the routing allocation changes by itself."
        ),
    )
    # Every probe here goes through a pod. A restart is precisely the thing an
    # external port-forward does not survive, and a probe that failed for that
    # reason would look exactly like the system failing to come back.
    del base_url
    payload = json.dumps({"applications": [APPLICATION]})
    endpoint = "http://inference-api/predict"

    alias_before = _production_version(config)
    identity_before = _identity_from_pod(cluster)
    traffic_before = cluster.configmap_value(API_CONFIGMAP, "ML_PLATFORM_CANARY_TRAFFIC_PERCENT")
    before = cluster.probe_from_pod(API_DEPLOYMENT, endpoint, payload, "restart-key")

    try:
        cluster.restart(PREDICTOR_DEPLOYMENT)
        cluster.wait_ready(PREDICTOR_DEPLOYMENT)
        cluster.restart(API_DEPLOYMENT)
        restarted = cluster.wait_ready(API_DEPLOYMENT)
        time.sleep(10)

        # The same routing key, so any tier change would show.
        after = cluster.probe_from_pod(API_DEPLOYMENT, endpoint, payload, "restart-key")
        identity_after = _identity_from_pod(cluster)
        traffic_after = cluster.configmap_value(API_CONFIGMAP, "ML_PLATFORM_CANARY_TRAFFIC_PERCENT")

        evidence.observations = {
            "before": before,
            "after": after,
            "identity_before": identity_before,
            "identity_after": identity_after,
            "traffic_before": traffic_before,
            "traffic_after": traffic_after,
        }
        evidence.observed_behaviour = (
            f"after restart /predict returned {after['status']} with "
            f"{after.get('probability')} (was {before.get('probability')}); "
            f"canary allocation {traffic_before}% -> {traffic_after}%"
        )
        evidence.blast_radius = [
            "requests fail or are refused during the rollout window",
            "in-process metric counters reset: they are per-process and not persisted",
        ]
        evidence.unaffected = [
            "the production alias",
            "the routing allocation, which is read from configuration rather than memory",
        ]
        evidence.recovered = bool(restarted and after["scored"])
        evidence.recovery_action = "none required; the rollout is the recovery"
        evidence.recovery_result = f"both deployments rolled out; serving resumed={after['scored']}"

        evidence.add(
            inv.recovery_returns_known_state(
                healthy=bool(restarted and after["scored"]),
                model_identity_before=identity_before,
                model_identity_after=identity_after,
            )
        )
        evidence.add(
            inv.InvariantResult(
                name="routing_state_unchanged_by_restart",
                held=traffic_before == traffic_after,
                detail=f"canary allocation {traffic_before}% before, {traffic_after}% after",
                evidence={"before": traffic_before, "after": traffic_after},
            )
        )
        # The score is the strongest identity check available: the same input
        # through the same model must give the same number.
        evidence.add(
            inv.InvariantResult(
                name="identical_input_gives_identical_score",
                held=before.get("probability") == after.get("probability"),
                detail=(
                    f"the same application scored {before.get('probability')} before and "
                    f"{after.get('probability')} after"
                ),
                evidence={"before": before.get("probability"), "after": after.get("probability")},
            )
        )
    finally:
        evidence.add(inv.production_alias_unchanged(alias_before, _production_version(config)))
        evidence.finish()
    return evidence


# --- scenario 3: a bad candidate (controlled double) ------------------------


def scenario_bad_candidate(config: Config) -> ScenarioEvidence:
    """Put a deliberately crippled candidate through the real promotion path.

    A controlled double, and labelled as one. The gates, the comparison and the
    registration code are the real ones; only the data and the registry are
    isolated, because destroying the live registry to prove that a bad model is
    rejected would be worse evidence, not better.
    """
    evidence = ScenarioEvidence(
        scenario="bad_candidate_promotion",
        description="A deliberately inferior candidate is offered for promotion.",
        mode=MODE_DOUBLE,
        failure_injected=(
            "a candidate trained with max_iter=1 and learning_rate=0.0001, offered through "
            "run_promotion with registration enabled"
        ),
        expected_behaviour=(
            "The M6 gates reject it, nothing is registered, and the production alias is "
            "unchanged. The rejection names the gates that failed."
        ),
    )
    from ml_platform.pipelines.promote_pipeline import run_promotion

    alias_before = _production_version(config)
    try:
        decision, _comparison = run_promotion(
            config=config,
            model_key="candidate",
            register=True,
            param_overrides={"max_iter": 1, "learning_rate": 0.0001, "max_depth": 1},
        )
        alias_after = _production_version(config)
        failures = [gate.name for gate in decision.report.failures]

        evidence.observations = {
            "promote": decision.report.promote,
            "registered": decision.registered,
            "failed_gates": failures,
            "gate_summary": decision.report.summary(),
            "alias_before": alias_before,
            "alias_after": alias_after,
        }
        evidence.observed_behaviour = (
            f"{decision.report.summary()}; failed gates: {failures or 'none'}"
        )
        evidence.blast_radius = [
            "none: the candidate keeps an ordinary MLflow run and never reaches the registry"
        ]
        evidence.unaffected = ["production", "serving", "the registry"]
        evidence.recovery_action = "none required; nothing was changed"
        evidence.recovery_result = "production is unchanged, so there is nothing to undo"
        evidence.recovered = True

        evidence.add(
            inv.failed_candidate_not_production(
                promoted=decision.report.promote,
                registered=bool(decision.registered),
                alias_before=alias_before,
                alias_after=alias_after,
            )
        )
        evidence.add(inv.production_alias_unchanged(alias_before, alias_after))
    except Exception as exc:
        evidence.error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("the bad-candidate scenario could not run", exc_info=True)
    finally:
        evidence.finish()
    return evidence
