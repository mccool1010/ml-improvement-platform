"""Tests for the Kubernetes manifests.

Same reasoning as :mod:`tests.unit.test_container` and
:mod:`tests.unit.test_ci_workflow`: these are configuration files, and the
mistakes worth catching are visible without a cluster. A deployment that has
quietly lost its readiness probe, started running as root, or had a host path
put back into it looks exactly like a working one until it is applied.

These tests read the YAML. They do not talk to a cluster.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from ml_platform.paths import project_root

MANIFEST_DIR = project_root() / "k8s" / "base"


@pytest.fixture(scope="module")
def manifests() -> list[dict[str, Any]]:
    """Every manifest in k8s/base, parsed."""
    documents: list[dict[str, Any]] = []
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if document:
                documents.append(document)
    return documents


def _by_kind(manifests: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [m for m in manifests if m["kind"] == kind]


def _named(manifests: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    for manifest in _by_kind(manifests, kind):
        if manifest["metadata"]["name"] == name:
            return manifest
    raise AssertionError(f"no {kind} named {name}")


@pytest.fixture(scope="module")
def api(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    return _named(manifests, "Deployment", "inference-api")


@pytest.fixture(scope="module")
def api_container(api: dict[str, Any]) -> dict[str, Any]:
    containers = api["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1, "one container per pod keeps the probes meaningful"
    return dict(containers[0])


class TestTheManifestsAreValid:
    def test_the_directory_exists(self) -> None:
        assert MANIFEST_DIR.is_dir()

    def test_every_manifest_parses(self, manifests: list[dict[str, Any]]) -> None:
        assert manifests

    def test_every_manifest_is_a_named_kubernetes_object(
        self, manifests: list[dict[str, Any]]
    ) -> None:
        for manifest in manifests:
            assert manifest.get("apiVersion"), manifest
            assert manifest.get("kind"), manifest
            assert manifest["metadata"]["name"], manifest

    def test_the_kustomization_lists_every_manifest(self) -> None:
        """A file that is not listed is silently never applied."""
        kustomization = yaml.safe_load(
            (MANIFEST_DIR / "kustomization.yaml").read_text(encoding="utf-8")
        )
        listed = set(kustomization["resources"])
        on_disk = {p.name for p in MANIFEST_DIR.glob("*.yaml")} - {"kustomization.yaml"}
        assert listed == on_disk

    def test_everything_is_in_one_namespace(self, manifests: list[dict[str, Any]]) -> None:
        for manifest in manifests:
            if manifest["kind"] == "Namespace":
                continue
            assert manifest["metadata"].get("namespace") == "ml-platform", manifest["metadata"]


class TestNothingNamesTheDeveloperMachine:
    """The M8 finding, as a test.

    The local MLflow store recorded an absolute Windows artifact path, which no
    pod can resolve. The fix was a tracking server that serves artifacts over
    HTTP. Nothing here may quietly reintroduce a host path.
    """

    def test_no_windows_path_appears(self) -> None:
        for path in MANIFEST_DIR.glob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            assert "C:/" not in text, path.name
            assert "C:\\" not in text, path.name

    def test_no_pod_mounts_a_host_path(self, manifests: list[dict[str, Any]]) -> None:
        for deployment in _by_kind(manifests, "Deployment"):
            for volume in deployment["spec"]["template"]["spec"].get("volumes", []):
                assert "hostPath" not in volume, f"{deployment['metadata']['name']}: {volume}"

    def test_the_tracking_server_serves_its_own_artifacts(
        self, manifests: list[dict[str, Any]]
    ) -> None:
        """Without --serve-artifacts, run artifact URIs are filesystem paths
        again and a client would need the server's disk."""
        mlflow = _named(manifests, "Deployment", "mlflow")
        command = mlflow["spec"]["template"]["spec"]["containers"][0]["command"]
        assert "--serve-artifacts" in command


class TestTheTrackingServerCanActuallyStart:
    """Every assertion here is a failure that happened on first deploy."""

    def _command(self, manifests: list[dict[str, Any]]) -> list[str]:
        return list(
            _named(manifests, "Deployment", "mlflow")["spec"]["template"]["spec"]["containers"][0][
                "command"
            ]
        )

    def test_the_worker_count_is_pinned(self, manifests: list[dict[str, Any]]) -> None:
        """`mlflow server` defaults to four workers, each importing the whole of
        MLflow. Four of them were OOMKilled before the server answered a probe."""
        assert any(c.startswith("--workers=") for c in self._command(manifests))

    def test_the_service_dns_names_are_allowed_hosts(self, manifests: list[dict[str, Any]]) -> None:
        """MLflow 3 validates the Host header against localhost and private IP
        literals by default, and refuses a Service DNS name as a rebinding
        attempt. Every in-cluster client sends exactly that."""
        allowed = [c for c in self._command(manifests) if c.startswith("--allowed-hosts=")]
        assert allowed, "the server would reject requests addressed to its Service"
        hosts = {h.strip() for h in allowed[0].split("=", 1)[1].split(",")}
        service = _named(manifests, "Service", "mlflow")
        name = service["metadata"]["name"]
        port = service["spec"]["ports"][0]["port"]
        assert f"{name}:{port}" in hosts, f"{name}:{port} is not an allowed host"

    def test_the_memory_limit_clears_the_observed_footprint(
        self, manifests: list[dict[str, Any]]
    ) -> None:
        container = _named(manifests, "Deployment", "mlflow")["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["resources"]["limits"]["memory"] == "2Gi"


class TestTheApiCannotHangOnStartup:
    """The API loads its model inside the startup lifespan. The MLflow client
    retries a connection failure with exponential backoff, so an unreachable
    tracking server left pods in `Waiting for application startup`, serving
    nothing, instead of starting and reporting themselves unready.
    """

    def test_the_model_load_is_time_bounded(self, manifests: list[dict[str, Any]]) -> None:
        config = _named(manifests, "ConfigMap", "inference-api-config")["data"]
        assert int(config["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) <= 3
        assert int(config["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 30

    def test_the_bound_is_shorter_than_the_startup_probe(
        self, manifests: list[dict[str, Any]], api_container: dict[str, Any]
    ) -> None:
        """Otherwise the probe kills the pod before the load can fail honestly,
        and the failure shows up as a restart loop instead of as `not ready`."""
        config = _named(manifests, "ConfigMap", "inference-api-config")["data"]
        worst_case = int(config["MLFLOW_HTTP_REQUEST_TIMEOUT"]) * (
            int(config["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) + 1
        )
        startup = api_container["startupProbe"]
        budget = startup["periodSeconds"] * startup["failureThreshold"]
        assert worst_case < budget, f"{worst_case}s load budget vs {budget}s probe budget"


class TestServiceDiscovery:
    def test_the_api_is_told_where_the_tracking_server_is(
        self, manifests: list[dict[str, Any]]
    ) -> None:
        config = _named(manifests, "ConfigMap", "inference-api-config")
        assert config["data"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"

    def test_that_address_is_a_service_that_exists(self, manifests: list[dict[str, Any]]) -> None:
        """The ConfigMap names `mlflow`; cluster DNS resolves it only if the
        Service is named that and listens on that port."""
        service = _named(manifests, "Service", "mlflow")
        assert service["spec"]["ports"][0]["port"] == 5000

    def test_the_api_reads_its_configuration_from_the_configmap(
        self, api_container: dict[str, Any]
    ) -> None:
        sources = [ref["configMapRef"]["name"] for ref in api_container.get("envFrom", [])]
        assert "inference-api-config" in sources

    def test_the_tracking_address_is_not_hardcoded_in_the_pod(
        self, api_container: dict[str, Any]
    ) -> None:
        """It must arrive as configuration, not be baked into the manifest."""
        inline = {entry["name"] for entry in api_container.get("env", [])}
        assert "MLFLOW_TRACKING_URI" not in inline

    def test_each_service_selects_its_own_deployment(self, manifests: list[dict[str, Any]]) -> None:
        for service in _by_kind(manifests, "Service"):
            selector = service["spec"]["selector"]
            matching = [
                d
                for d in _by_kind(manifests, "Deployment")
                if d["spec"]["selector"]["matchLabels"] == selector
            ]
            assert len(matching) == 1, f"{service['metadata']['name']} selects {len(matching)}"

    def test_the_api_service_is_internal_only(self, manifests: list[dict[str, Any]]) -> None:
        """Exposure is a decision to take deliberately, not inherit."""
        for service in _by_kind(manifests, "Service"):
            assert service["spec"]["type"] == "ClusterIP", service["metadata"]["name"]


class TestHealthManagement:
    def test_liveness_uses_the_health_endpoint(self, api_container: dict[str, Any]) -> None:
        assert api_container["livenessProbe"]["httpGet"]["path"] == "/health"

    def test_readiness_uses_the_ready_endpoint(self, api_container: dict[str, Any]) -> None:
        assert api_container["readinessProbe"]["httpGet"]["path"] == "/ready"

    def test_liveness_does_not_ask_readiness(self, api_container: dict[str, Any]) -> None:
        """A pod holding no promoted model is working correctly. Restarting it in
        a loop would neither produce a model nor help anyone."""
        assert api_container["livenessProbe"]["httpGet"]["path"] != "/ready"

    def test_a_startup_probe_covers_the_model_load(self, api_container: dict[str, Any]) -> None:
        """The registry lookup and artifact download are slow and variable, and
        must not be counted against liveness."""
        startup = api_container["startupProbe"]
        assert startup["httpGet"]["path"] == "/health"
        assert startup["periodSeconds"] * startup["failureThreshold"] >= 60

    def test_every_probe_targets_the_named_container_port(
        self, api_container: dict[str, Any]
    ) -> None:
        port_names = {port["name"] for port in api_container["ports"]}
        for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
            assert api_container[probe]["httpGet"]["port"] in port_names

    def test_the_tracking_server_is_probed_too(self, manifests: list[dict[str, Any]]) -> None:
        container = _named(manifests, "Deployment", "mlflow")["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        assert container["readinessProbe"]["httpGet"]["path"] == "/health"


class TestRuntimeSafety:
    def test_the_api_does_not_run_as_root(self, api: dict[str, Any]) -> None:
        security = api["spec"]["template"]["spec"]["securityContext"]
        assert security["runAsNonRoot"] is True
        assert security["runAsUser"] == 10001

    def test_the_uid_matches_the_image(self, api: dict[str, Any]) -> None:
        """docker/Dockerfile creates appuser with uid 10001. A mismatch would
        leave the pod unable to read files the image owns."""
        dockerfile = (project_root() / "docker" / "Dockerfile").read_text(encoding="utf-8")
        assert "--uid 10001" in dockerfile
        assert api["spec"]["template"]["spec"]["securityContext"]["runAsUser"] == 10001

    def test_privileges_cannot_be_escalated(self, api_container: dict[str, Any]) -> None:
        assert api_container["securityContext"]["allowPrivilegeEscalation"] is False

    def test_all_capabilities_are_dropped(self, api_container: dict[str, Any]) -> None:
        assert api_container["securityContext"]["capabilities"]["drop"] == ["ALL"]

    def test_every_deployment_runs_as_non_root(self, manifests: list[dict[str, Any]]) -> None:
        for deployment in _by_kind(manifests, "Deployment"):
            security = deployment["spec"]["template"]["spec"]["securityContext"]
            assert security["runAsNonRoot"] is True, deployment["metadata"]["name"]


class TestResources:
    @pytest.mark.parametrize("deployment_name", ["inference-api", "mlflow"])
    def test_requests_and_limits_are_declared(
        self, manifests: list[dict[str, Any]], deployment_name: str
    ) -> None:
        """Without a request the scheduler cannot place the pod honestly; without
        a limit one pod can take the node down with it."""
        container = _named(manifests, "Deployment", deployment_name)["spec"]["template"]["spec"][
            "containers"
        ][0]
        resources = container["resources"]
        for section in ("requests", "limits"):
            assert set(resources[section]) == {"cpu", "memory"}, f"{deployment_name}.{section}"

    def test_the_image_is_never_pulled(self, manifests: list[dict[str, Any]]) -> None:
        """It is built locally and never pushed, so Always would fail the pull."""
        for deployment in _by_kind(manifests, "Deployment"):
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            assert container["imagePullPolicy"] == "IfNotPresent"

    def test_both_workloads_run_the_same_image(self, manifests: list[dict[str, Any]]) -> None:
        """The tracking server runs the project image so the MLflow build that
        writes the registry is the one from uv.lock that reads it."""
        images = {
            d["spec"]["template"]["spec"]["containers"][0]["image"]
            for d in _by_kind(manifests, "Deployment")
        }
        assert len(images) == 1, images

    def test_the_tracking_server_is_not_rolled_over_itself(
        self, manifests: list[dict[str, Any]]
    ) -> None:
        """One SQLite writer, one ReadWriteOnce volume. A rolling update would
        briefly run two pods against both."""
        mlflow = _named(manifests, "Deployment", "mlflow")
        assert mlflow["spec"]["strategy"]["type"] == "Recreate"
        assert mlflow["spec"]["replicas"] == 1
