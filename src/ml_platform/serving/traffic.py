"""Making a traffic change take effect where requests actually arrive.

A :class:`~ml_platform.serving.canary.CanaryRouter` lives inside one process. A
CLI that builds a router, stops it and exits has rolled back nothing: the API
replicas serving real traffic never saw it. That was a real defect in the first
cut of M14, and this module is the fix.

The allocation the replicas honour comes from their configuration -- the
``ML_PLATFORM_CANARY_TRAFFIC_PERCENT`` key the deployment sets -- so changing
traffic means changing that, which is the mechanism M14 already chose so that
every replica agrees by construction. A controller writes it.

Two implementations, because the two callers are genuinely different:

:class:`InProcessTrafficController` moves the router this process holds. That is
what the API itself would use, and what the tests use; it takes effect on the
next request with no restart.

:class:`KubernetesTrafficController` patches the ConfigMap and rolls the
Deployment, so replicas pick the new allocation up. It is slower -- a rollout,
not an assignment -- and that cost is stated rather than hidden.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.serving.canary import CanaryRouter

LOGGER = logging.getLogger(__name__)

#: The ConfigMap key the API reads its allocation from. Must match the env var
#: name in k8s/base/20-api-configmap.yaml and ml_platform.config.
TRAFFIC_KEY = "ML_PLATFORM_CANARY_TRAFFIC_PERCENT"

DEFAULT_TIMEOUT_SECONDS = 120


class TrafficControlError(RuntimeError):
    """Raised when a traffic change could not be applied."""


class TrafficController(Protocol):
    """Something that can change where live traffic goes."""

    def set_traffic(self, percent: float) -> None: ...

    def describe(self) -> str: ...


class InProcessTrafficController:
    """Moves the router held by this process.

    Immediate: the next request routes by the new allocation. This is the path a
    rollback triggered from inside the API would take, and the one the tests
    exercise.
    """

    def __init__(self, router: CanaryRouter) -> None:
        self._router = router

    def set_traffic(self, percent: float) -> None:
        if percent <= 0:
            self._router.stop("traffic set to zero")
        else:
            self._router.set_traffic(percent)
        _publish(self._router.traffic_percent)

    def describe(self) -> str:
        return "in-process router"


class KubernetesTrafficController:
    """Changes the allocation every API replica reads, then rolls them.

    Two steps, and both are needed. Patching the ConfigMap alone changes nothing
    for a running pod: the value is read into the router once at startup, so
    pods already serving keep the old split until they restart. The rollout is
    what makes the change real.

    ``kubectl rollout status`` is waited on, so the caller learns that the
    rollback landed rather than assuming it.
    """

    def __init__(
        self,
        namespace: str = "ml-platform",
        configmap: str = "inference-api-config",
        deployment: str = "inference-api",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.namespace = namespace
        self.configmap = configmap
        self.deployment = deployment
        self.timeout_seconds = timeout_seconds

    def _kubectl(self, *args: str) -> str:
        binary = shutil.which("kubectl")
        if binary is None:
            raise TrafficControlError(
                "kubectl is not on PATH, so the live allocation cannot be changed. "
                "Set the ConfigMap key by hand and restart the deployment."
            )
        try:
            result = subprocess.run(
                [binary, "-n", self.namespace, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TrafficControlError(
                f"kubectl {' '.join(args)} failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TrafficControlError(f"kubectl {' '.join(args)} timed out") from exc
        return result.stdout.strip()

    def set_traffic(self, percent: float) -> None:
        """Write the allocation and wait for the replicas to be serving it."""
        patch = f'{{"data":{{"{TRAFFIC_KEY}":"{percent}"}}}}'
        self._kubectl("patch", "configmap", self.configmap, "--type", "merge", "-p", patch)
        LOGGER.info("set %s=%s in configmap/%s", TRAFFIC_KEY, percent, self.configmap)

        # A ConfigMap change is invisible to a running pod here: the allocation
        # is read once at startup. The restart is what applies it.
        self._kubectl("rollout", "restart", f"deployment/{self.deployment}")
        self._kubectl(
            "rollout",
            "status",
            f"deployment/{self.deployment}",
            f"--timeout={self.timeout_seconds}s",
        )
        LOGGER.info("deployment/%s is serving %.2f%% canary traffic", self.deployment, percent)
        _publish(percent)

    def current_traffic(self) -> float:
        """What the ConfigMap currently says, for verification."""
        value = self._kubectl(
            "get", "configmap", self.configmap, "-o", f"jsonpath={{.data.{TRAFFIC_KEY}}}"
        )
        return float(value) if value else 0.0

    def describe(self) -> str:
        return f"configmap/{self.configmap} + deployment/{self.deployment} in {self.namespace}"


def _publish(percent: float) -> None:
    from ml_platform.observability.metrics import set_canary_traffic

    set_canary_traffic(percent)
