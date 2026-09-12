"""Breaking and repairing real components, through kubectl.

The smallest thing that can inject a real failure: scale a Deployment to zero,
scale it back, restart it, and ask what is running. No chaos framework, no
operator, no CRD -- those would be a large dependency to prove a small point, and
this project's whole argument is against that.

Every injector has a matching restore, and the scenario runner calls it in a
``finally``. A failure harness that can leave the cluster broken is a worse
liability than the failures it is testing for.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "ml-platform"
DEFAULT_TIMEOUT = 300

#: Run inside an API pod by :meth:`ClusterControl.probe_from_pod`. Kept as a
#: module constant so it is readable rather than assembled from fragments.
PROBE_SCRIPT = """
import json, urllib.request, urllib.error
request = urllib.request.Request(
    {url},
    data={payload}.encode(),
    headers={{"content-type": "application/json", "x-ml-routing-key": {key}}},
)
try:
    response = urllib.request.urlopen(request, timeout=30)
    body = json.load(response)
    predictions = body.get("predictions") or []
    print(json.dumps({{
        "status": response.status,
        "scored": bool(predictions),
        "probability": predictions[0]["default_probability"] if predictions else None,
        "tier": body.get("model", {{}}).get("serving_tier"),
    }}))
except urllib.error.HTTPError as exc:
    print(json.dumps({{
        "status": exc.code,
        "scored": False,
        "detail": exc.read().decode("utf-8", "replace")[:200],
    }}))
except Exception as exc:
    print(json.dumps({{"status": None, "scored": False, "detail": str(exc)[:200]}}))
"""

READY_SCRIPT = """
import json, urllib.request, urllib.error
try:
    response = urllib.request.urlopen({url}, timeout=20)
    print(json.dumps({{"status": response.status, "body": response.read().decode()[:600]}}))
except urllib.error.HTTPError as exc:
    print(json.dumps({{"status": exc.code, "body": exc.read().decode("utf-8", "replace")[:600]}}))
except Exception as exc:
    print(json.dumps({{"status": None, "body": str(exc)[:200]}}))
"""


class ClusterError(RuntimeError):
    """Raised when a cluster operation could not be carried out."""


@dataclass
class ScaleState:
    """What a Deployment was scaled to before a scenario touched it."""

    deployment: str
    replicas: int


class ClusterControl:
    """Scale, restart and inspect Deployments in one namespace."""

    def __init__(self, namespace: str = DEFAULT_NAMESPACE, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.namespace = namespace
        self.timeout = timeout

    # --- plumbing ----------------------------------------------------------

    def _kubectl(self, *args: str, check: bool = True) -> str:
        binary = shutil.which("kubectl")
        if binary is None:
            raise ClusterError("kubectl is not on PATH; live scenarios cannot run")
        try:
            result = subprocess.run(
                [binary, "-n", self.namespace, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=check,
            )
        except subprocess.CalledProcessError as exc:
            raise ClusterError(
                f"kubectl {' '.join(args)}: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClusterError(f"kubectl {' '.join(args)} timed out") from exc
        return result.stdout.strip()

    def available(self) -> bool:
        """Whether a cluster is reachable at all."""
        try:
            self._kubectl("get", "namespace", self.namespace)
            return True
        except ClusterError:
            return False

    # --- observation -------------------------------------------------------

    def replicas(self, deployment: str) -> int:
        value = self._kubectl("get", "deployment", deployment, "-o", "jsonpath={.spec.replicas}")
        return int(value) if value else 0

    def ready_replicas(self, deployment: str) -> int:
        value = self._kubectl(
            "get", "deployment", deployment, "-o", "jsonpath={.status.readyReplicas}"
        )
        return int(value) if value else 0

    def snapshot(self, deployments: list[str]) -> dict[str, dict[str, int]]:
        """Desired and ready replicas, for a before/after comparison."""
        return {
            name: {"desired": self.replicas(name), "ready": self.ready_replicas(name)}
            for name in deployments
        }

    # --- injection and repair ---------------------------------------------

    def scale(self, deployment: str, replicas: int) -> ScaleState:
        """Scale a Deployment, returning what it was, so it can be put back."""
        previous = self.replicas(deployment)
        self._kubectl("scale", f"deployment/{deployment}", f"--replicas={replicas}")
        LOGGER.info("scaled deployment/%s from %d to %d", deployment, previous, replicas)
        return ScaleState(deployment=deployment, replicas=previous)

    def restore(self, state: ScaleState) -> None:
        self._kubectl("scale", f"deployment/{state.deployment}", f"--replicas={state.replicas}")
        LOGGER.info("restored deployment/%s to %d replica(s)", state.deployment, state.replicas)

    def restart(self, deployment: str) -> None:
        self._kubectl("rollout", "restart", f"deployment/{deployment}")

    def wait_ready(self, deployment: str, *, timeout: int | None = None) -> bool:
        """Wait for a rollout to finish. Returns whether it did."""
        try:
            self._kubectl(
                "rollout",
                "status",
                f"deployment/{deployment}",
                f"--timeout={timeout or self.timeout}s",
            )
            return True
        except ClusterError:
            return False

    def wait_gone(self, deployment: str, *, attempts: int = 30, delay: float = 2.0) -> bool:
        """Wait until a Deployment has no ready replicas.

        Scaling to zero returns immediately; the pod takes a moment to go. A
        scenario that started observing before that would measure the healthy
        system and conclude, wrongly, that nothing broke.
        """
        for _ in range(attempts):
            if self.ready_replicas(deployment) == 0:
                return True
            time.sleep(delay)
        return False

    def wait_serving(self, deployment: str, *, attempts: int = 60, delay: float = 2.0) -> bool:
        for _ in range(attempts):
            if self.ready_replicas(deployment) >= 1:
                return True
            time.sleep(delay)
        return False

    def patch_configmap(self, name: str, data: dict[str, str]) -> None:
        import json

        self._kubectl(
            "patch", "configmap", name, "--type", "merge", "-p", json.dumps({"data": data})
        )

    def configmap_value(self, name: str, key: str) -> str:
        return self._kubectl("get", "configmap", name, "-o", f"jsonpath={{.data.{key}}}")

    def probe_from_pod(self, deployment: str, url: str, payload: str, key: str) -> dict[str, Any]:
        """Send one prediction from inside the cluster, through the Service.

        A port-forward is the obvious way to probe, and it is the wrong one for
        any scenario that restarts a Deployment: the forward dies with the pod
        it was attached to, so every subsequent probe fails as a connection
        error, indistinguishable from the failure being studied. The first run
        of the canary scenario recorded 40 of 40 requests failing for exactly
        that reason, and the number looked like evidence.

        Running the probe inside a pod removes the moving part. The request
        still crosses the real Service, so it is real traffic, not a simulation.
        """
        import json as _json

        script = PROBE_SCRIPT.format(
            url=_json.dumps(url), payload=_json.dumps(payload), key=_json.dumps(key)
        )
        try:
            output = self._kubectl("exec", f"deployment/{deployment}", "--", "python", "-c", script)
            return dict(_json.loads(output.strip().splitlines()[-1]))
        except Exception as exc:
            return {"status": None, "scored": False, "detail": f"probe failed: {exc}"}

    def ready_from_pod(self, deployment: str, url: str) -> dict[str, Any]:
        """Read /ready from inside the cluster, for the same reason as above."""
        import json as _json

        script = READY_SCRIPT.format(url=_json.dumps(url))
        try:
            output = self._kubectl("exec", f"deployment/{deployment}", "--", "python", "-c", script)
            return dict(_json.loads(output.strip().splitlines()[-1]))
        except Exception as exc:
            return {"status": None, "detail": f"probe failed: {exc}"}

    def describe(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "kubectl": shutil.which("kubectl") is not None}
