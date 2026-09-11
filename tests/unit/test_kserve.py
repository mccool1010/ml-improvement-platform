"""Tests for the KServe manifests.

Same reasoning as :mod:`tests.unit.test_kubernetes`: these are configuration
files, and the mistakes worth catching are visible without a cluster.

The ones that matter here are about the arrangement ADR-002 chose. Two things
serving the same model, a storage URI that quietly names a developer's machine,
or a model name that does not match the InferenceService, all look like working
manifests right up until they are applied.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from ml_platform.paths import project_root

KSERVE_DIR = project_root() / "k8s" / "kserve"
BASE_DIR = project_root() / "k8s" / "base"


def _load(directory: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if document:
                documents.append(document)
    return documents


def _named(manifests: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    for manifest in manifests:
        if manifest["kind"] == kind and manifest["metadata"]["name"] == name:
            return manifest
    raise AssertionError(f"no {kind} named {name}")


@pytest.fixture(scope="module")
def kserve() -> list[dict[str, Any]]:
    return _load(KSERVE_DIR)


@pytest.fixture(scope="module")
def base() -> list[dict[str, Any]]:
    return _load(BASE_DIR)


@pytest.fixture(scope="module")
def isvc(kserve: list[dict[str, Any]]) -> dict[str, Any]:
    return _named(kserve, "InferenceService", "sba-loan-default")


@pytest.fixture(scope="module")
def runtime(kserve: list[dict[str, Any]]) -> dict[str, Any]:
    return _named(kserve, "ServingRuntime", "mlflow-lockfile-runtime")


@pytest.fixture(scope="module")
def runtime_env(runtime: dict[str, Any]) -> dict[str, str]:
    return {e["name"]: e["value"] for e in runtime["spec"]["containers"][0]["env"]}


class TestTheManifestsAreValid:
    def test_they_parse(self, kserve: list[dict[str, Any]]) -> None:
        assert kserve

    def test_the_kustomization_lists_every_manifest(self) -> None:
        """A file that is not listed is silently never applied."""
        kustomization = yaml.safe_load(
            (KSERVE_DIR / "kustomization.yaml").read_text(encoding="utf-8")
        )
        listed = set(kustomization["resources"])
        on_disk = {p.name for p in KSERVE_DIR.glob("*.yaml")} - {"kustomization.yaml"}
        assert listed == on_disk

    def test_everything_shares_the_namespace(self, kserve: list[dict[str, Any]]) -> None:
        for manifest in kserve:
            assert manifest["metadata"]["namespace"] == "ml-platform", manifest["metadata"]["name"]

    def test_raw_deployment_mode_is_requested(self, isvc: dict[str, Any]) -> None:
        """There is no Knative and no service mesh here. Serverless mode would
        leave the InferenceService waiting for both indefinitely."""
        annotations = isvc["metadata"]["annotations"]
        assert annotations["serving.kserve.io/deploymentMode"] == "RawDeployment"


class TestThereIsOneProductionServingPath:
    """ADR-002: KServe is canonical, FastAPI is the application tier in front of
    it. Two things serving one model drift apart, and a canary proven on one
    says nothing about the other."""

    def test_the_application_tier_points_at_the_model_tier(
        self, base: list[dict[str, Any]], isvc: dict[str, Any]
    ) -> None:
        config = _named(base, "ConfigMap", "inference-api-config")["data"]
        # KServe names the predictor Service after the InferenceService.
        assert config["ML_PLATFORM_PREDICTOR_URL"] == f"http://{isvc['metadata']['name']}-predictor"

    def test_the_application_tier_does_not_also_serve_the_model(
        self, base: list[dict[str, Any]]
    ) -> None:
        container = _named(base, "Deployment", "inference-api")["spec"]["template"]["spec"][
            "containers"
        ][0]
        command = " ".join(container.get("command", []) or [])
        assert "ml_platform.serving.predictor" not in command

    def test_both_tiers_run_the_same_image(
        self, base: list[dict[str, Any]], runtime: dict[str, Any]
    ) -> None:
        """The environment that produced the artifact is the one that loads it."""
        api = _named(base, "Deployment", "inference-api")["spec"]["template"]["spec"]["containers"][
            0
        ]
        assert runtime["spec"]["containers"][0]["image"] == api["image"]


class TestTheModelIsTraceableToTheRegistry:
    def test_the_storage_uri_is_a_cluster_volume(self, isvc: dict[str, Any]) -> None:
        """Not a host path. M8 established that the local store's absolute
        artifact URI is unreachable from a pod."""
        uri = isvc["spec"]["predictor"]["model"]["storageUri"]
        assert uri.startswith("pvc://")
        assert "C:/" not in uri
        assert "C:" + chr(92) not in uri

    def test_it_names_the_tracking_servers_own_volume(
        self, isvc: dict[str, Any], base: list[dict[str, Any]]
    ) -> None:
        """A different claim would serve whatever happened to be on it."""
        claim = _named(base, "PersistentVolumeClaim", "mlflow-store")["metadata"]["name"]
        assert isvc["spec"]["predictor"]["model"]["storageUri"].startswith(f"pvc://{claim}/")

    def test_the_version_it_serves_is_recorded(self, isvc: dict[str, Any]) -> None:
        """The path carries an MLflow run id. The annotations say which
        registered version that is, so a reader never has to decode the path."""
        annotations = isvc["metadata"]["annotations"]
        assert annotations["ml-platform.io/registry-alias"] == "production"
        assert annotations["ml-platform.io/registered-model"]
        assert annotations["ml-platform.io/model-version"]

    def test_the_derivation_script_agrees_with_the_manifest(self, isvc: dict[str, Any]) -> None:
        """The script translates a registry source into this path. If the two
        disagree, the manifest is hand-maintained and can name any model."""
        import sys

        sys.path.insert(0, str(project_root() / "scripts"))
        from kserve_model_uri import storage_uri

        uri = isvc["spec"]["predictor"]["model"]["storageUri"]
        run_id = uri.rstrip("/").split("/")[-3]
        assert storage_uri(f"mlflow-artifacts:/1/{run_id}/artifacts/model", "mlflow-store") == uri

    def test_a_local_file_source_is_refused(self) -> None:
        """The M8 failure, as a test: a file:// source is one machine's path and
        cannot become a cluster volume reference."""
        import sys

        sys.path.insert(0, str(project_root() / "scripts"))
        from kserve_model_uri import ResolutionError, storage_uri

        with pytest.raises(ResolutionError, match="not served by the tracking server"):
            storage_uri("file:///C:/mlops/mlartifacts/1/abc/artifacts/model", "mlflow-store")


class TestTheModelTier:
    def test_the_runtime_is_not_auto_selected(self, runtime: dict[str, Any]) -> None:
        """A model must name this runtime, so nothing lands here by accident."""
        for fmt in runtime["spec"]["supportedModelFormats"]:
            assert fmt.get("autoSelect") is False

    def test_the_inference_service_names_the_runtime(self, isvc: dict[str, Any]) -> None:
        assert isvc["spec"]["predictor"]["model"]["runtime"] == "mlflow-lockfile-runtime"

    def test_it_runs_the_projects_predictor(self, runtime: dict[str, Any]) -> None:
        """`mlflow models serve` answers with hard labels here: the artifact
        records predict_fn: predict, and this project decides on a probability."""
        command = " ".join(runtime["spec"]["containers"][0]["command"])
        assert "ml_platform.serving.predictor" in command

    def test_the_feature_set_is_stated(self, runtime_env: dict[str, str]) -> None:
        """Serving a different representation than the model was trained on is
        the skew a single scoring path exists to prevent."""
        assert runtime_env["MODEL_FEATURE_SET"]

    def test_the_model_name_matches_the_inference_service(
        self, runtime_env: dict[str, str], isvc: dict[str, Any]
    ) -> None:
        """V1 addresses a named model and the predictor refuses a name it does
        not serve, so a mismatch is a 404 on every single request."""
        assert runtime_env["MODEL_NAME"] == isvc["metadata"]["name"]

    def test_native_thread_pools_are_pinned(self, runtime_env: dict[str, str]) -> None:
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert runtime_env[variable] == "1"

    def test_it_does_not_run_as_root(self, runtime: dict[str, Any]) -> None:
        security = runtime["spec"]["containers"][0]["securityContext"]
        assert security["runAsNonRoot"] is True
        assert security["runAsUser"] == 10001
        assert security["allowPrivilegeEscalation"] is False
        assert security["capabilities"]["drop"] == ["ALL"]

    @pytest.mark.parametrize("probe", ["readinessProbe", "livenessProbe", "startupProbe"])
    def test_probes_ask_health(self, isvc: dict[str, Any], probe: str) -> None:
        assert isvc["spec"]["predictor"]["model"][probe]["httpGet"]["path"] == "/health"

    def test_startup_is_not_counted_against_liveness(self, isvc: dict[str, Any]) -> None:
        """Deserialising the pipeline takes far longer than answering a request,
        and that is not a liveness failure."""
        startup = isvc["spec"]["predictor"]["model"]["startupProbe"]
        assert startup["periodSeconds"] * startup["failureThreshold"] >= 120

    def test_resources_are_declared(self, isvc: dict[str, Any]) -> None:
        resources = isvc["spec"]["predictor"]["model"]["resources"]
        for section in ("requests", "limits"):
            assert set(resources[section]) == {"cpu", "memory"}, section
