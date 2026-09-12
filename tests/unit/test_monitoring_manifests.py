"""Tests for the observability manifests.

Same reasoning as the other manifest tests: a scrape annotation that has gone
missing, a dashboard whose data source uid no longer matches, or a trace exporter
pointed at a Service that does not exist all look like working configuration
until someone opens a dashboard and finds it empty.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from ml_platform.paths import project_root

MONITORING_DIR = project_root() / "k8s" / "monitoring"
BASE_DIR = project_root() / "k8s" / "base"
KSERVE_DIR = project_root() / "k8s" / "kserve"


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
def monitoring() -> list[dict[str, Any]]:
    return _load(MONITORING_DIR)


@pytest.fixture(scope="module")
def base() -> list[dict[str, Any]]:
    return _load(BASE_DIR)


@pytest.fixture(scope="module")
def kserve() -> list[dict[str, Any]]:
    return _load(KSERVE_DIR)


@pytest.fixture(scope="module")
def dashboard(monitoring: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _named(monitoring, "ConfigMap", "grafana-dashboard-ml-platform")["data"][
        "ml-platform.json"
    ]
    loaded: dict[str, Any] = json.loads(raw)
    return loaded


class TestTheManifestsAreValid:
    def test_they_parse(self, monitoring: list[dict[str, Any]]) -> None:
        assert monitoring

    def test_the_kustomization_lists_every_manifest(self) -> None:
        kustomization = yaml.safe_load(
            (MONITORING_DIR / "kustomization.yaml").read_text(encoding="utf-8")
        )
        listed = set(kustomization["resources"])
        on_disk = {p.name for p in MONITORING_DIR.glob("*.yaml")} - {"kustomization.yaml"}
        assert listed == on_disk

    def test_everything_shares_the_namespace(self, monitoring: list[dict[str, Any]]) -> None:
        cluster_scoped = {"ClusterRole", "ClusterRoleBinding"}
        for manifest in monitoring:
            if manifest["kind"] in cluster_scoped:
                continue
            assert manifest["metadata"]["namespace"] == "ml-platform", manifest["metadata"]["name"]

    @pytest.mark.parametrize("name", ["prometheus", "grafana", "jaeger"])
    def test_each_component_has_a_service(
        self, monitoring: list[dict[str, Any]], name: str
    ) -> None:
        """Service discovery by DNS name, like everything else here."""
        assert _named(monitoring, "Service", name)["spec"]["type"] == "ClusterIP"

    @pytest.mark.parametrize("name", ["prometheus", "grafana", "jaeger"])
    def test_each_component_is_probed(self, monitoring: list[dict[str, Any]], name: str) -> None:
        container = _named(monitoring, "Deployment", name)["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["readinessProbe"]["httpGet"]["path"]
        assert container["livenessProbe"]["httpGet"]["path"]

    @pytest.mark.parametrize("name", ["prometheus", "grafana", "jaeger"])
    def test_each_component_declares_resources(
        self, monitoring: list[dict[str, Any]], name: str
    ) -> None:
        """Docker Desktop has one node's worth of memory to share."""
        resources = _named(monitoring, "Deployment", name)["spec"]["template"]["spec"][
            "containers"
        ][0]["resources"]
        for section in ("requests", "limits"):
            assert set(resources[section]) == {"cpu", "memory"}, f"{name}.{section}"

    @pytest.mark.parametrize("name", ["prometheus", "grafana", "jaeger"])
    def test_each_component_runs_as_non_root(
        self, monitoring: list[dict[str, Any]], name: str
    ) -> None:
        security = _named(monitoring, "Deployment", name)["spec"]["template"]["spec"][
            "securityContext"
        ]
        assert security["runAsNonRoot"] is True, name


class TestPrometheusCanActuallyScrape:
    def test_it_has_a_service_account(self, monitoring: list[dict[str, Any]]) -> None:
        """Pod discovery is an API call; without RBAC every target disappears."""
        deployment = _named(monitoring, "Deployment", "prometheus")
        assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == "prometheus"
        _named(monitoring, "ServiceAccount", "prometheus")

    def test_its_permissions_are_read_only(self, monitoring: list[dict[str, Any]]) -> None:
        """A scraper has no business changing anything."""
        role = _named(monitoring, "ClusterRole", "ml-platform-prometheus")
        for rule in role["rules"]:
            assert set(rule["verbs"]) <= {"get", "list", "watch"}, rule

    def test_discovery_is_by_annotation(self, monitoring: list[dict[str, Any]]) -> None:
        config = yaml.safe_load(
            _named(monitoring, "ConfigMap", "prometheus-config")["data"]["prometheus.yml"]
        )
        job = config["scrape_configs"][0]
        assert job["kubernetes_sd_configs"][0]["role"] == "pod"
        sources = [
            source for rule in job["relabel_configs"] for source in rule.get("source_labels", [])
        ]
        assert "__meta_kubernetes_pod_annotation_prometheus_io_scrape" in sources

    def test_the_application_tier_advertises_itself(self, base: list[dict[str, Any]]) -> None:
        annotations = _named(base, "Deployment", "inference-api")["spec"]["template"]["metadata"][
            "annotations"
        ]
        assert annotations["prometheus.io/scrape"] == "true"
        assert annotations["prometheus.io/path"] == "/metrics"

    def test_the_scrape_port_matches_the_container_port(self, base: list[dict[str, Any]]) -> None:
        """A mismatch produces a target that is up in the config and down in fact."""
        template = _named(base, "Deployment", "inference-api")["spec"]["template"]
        port = template["spec"]["containers"][0]["ports"][0]["containerPort"]
        assert template["metadata"]["annotations"]["prometheus.io/port"] == str(port)

    def test_the_model_tier_advertises_itself(self, kserve: list[dict[str, Any]]) -> None:
        """KServe writes that Deployment, so the annotation has to travel on the
        InferenceService for the pod to be scraped at all."""
        predictor = _named(kserve, "InferenceService", "sba-loan-default")["spec"]["predictor"]
        assert predictor["annotations"]["prometheus.io/scrape"] == "true"
        assert predictor["annotations"]["prometheus.io/port"] == "8080"


class TestTracingIsWiredEndToEnd:
    def test_jaeger_accepts_otlp(self, monitoring: list[dict[str, Any]]) -> None:
        """There is no separate collector; the application exports straight here."""
        container = _named(monitoring, "Deployment", "jaeger")["spec"]["template"]["spec"][
            "containers"
        ][0]
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["COLLECTOR_OTLP_ENABLED"] == "true"
        ports = {p["name"]: p["containerPort"] for p in container["ports"]}
        assert ports["otlp-http"] == 4318

    def test_both_tiers_export_to_the_jaeger_service(
        self,
        base: list[dict[str, Any]],
        kserve: list[dict[str, Any]],
        monitoring: list[dict[str, Any]],
    ) -> None:
        """Both halves of a trace must land in the same backend, or the trace is
        two unrelated fragments."""
        service = _named(monitoring, "Service", "jaeger")
        otlp_port = next(p["port"] for p in service["spec"]["ports"] if p["name"] == "otlp-http")
        expected = f"http://{service['metadata']['name']}:{otlp_port}"

        api = _named(base, "ConfigMap", "inference-api-config")["data"]
        assert api["OTEL_EXPORTER_OTLP_ENDPOINT"] == expected

        model_env = {
            e["name"]: e["value"]
            for e in _named(kserve, "InferenceService", "sba-loan-default")["spec"]["predictor"][
                "model"
            ]["env"]
        }
        assert model_env["OTEL_EXPORTER_OTLP_ENDPOINT"] == expected


class TestTheDashboardIsProvisioned:
    def test_it_is_valid_json(self, dashboard: dict[str, Any]) -> None:
        assert dashboard["title"]
        assert dashboard["panels"]

    def test_grafana_is_given_a_provider_and_a_datasource(
        self, monitoring: list[dict[str, Any]]
    ) -> None:
        """Provisioned from files, so a fresh cluster has the dashboard without
        anyone rebuilding it from memory."""
        provisioning = _named(monitoring, "ConfigMap", "grafana-provisioning")["data"]
        assert "datasources.yaml" in provisioning
        assert "dashboards.yaml" in provisioning

    def test_the_datasource_points_at_the_prometheus_service(
        self, monitoring: list[dict[str, Any]]
    ) -> None:
        datasources = yaml.safe_load(
            _named(monitoring, "ConfigMap", "grafana-provisioning")["data"]["datasources.yaml"]
        )
        source = datasources["datasources"][0]
        service = _named(monitoring, "Service", "prometheus")
        port = service["spec"]["ports"][0]["port"]
        assert source["url"] == f"http://{service['metadata']['name']}:{port}"

    def test_every_panel_uses_the_provisioned_datasource_uid(
        self, dashboard: dict[str, Any], monitoring: list[dict[str, Any]]
    ) -> None:
        """A uid mismatch is the classic way a provisioned dashboard comes up
        entirely blank."""
        datasources = yaml.safe_load(
            _named(monitoring, "ConfigMap", "grafana-provisioning")["data"]["datasources.yaml"]
        )
        uid = datasources["datasources"][0]["uid"]
        for panel in dashboard["panels"]:
            assert panel["datasource"]["uid"] == uid, panel["title"]

    def test_the_required_panels_are_present(self, dashboard: dict[str, Any]) -> None:
        titles = " ".join(panel["title"].lower() for panel in dashboard["panels"])
        for subject in ("request rate", "error", "latency", "prediction", "model tier"):
            assert subject in titles, subject

    def test_every_query_names_a_metric_the_code_exports(self, dashboard: dict[str, Any]) -> None:
        """A dashboard querying a metric nobody emits is an empty panel that
        looks like a healthy system."""
        from prometheus_client import REGISTRY

        # Importing the module is what registers the metrics; without it the
        # comparison below would pass vacuously against an empty set.
        from ml_platform.observability import metrics  # noqa: F401

        exported = {
            metric.name for metric in REGISTRY.collect() if metric.name.startswith("ml_platform_")
        }
        assert exported, "no ml_platform metrics are registered; the check would be vacuous"
        for panel in dashboard["panels"]:
            for target in panel["targets"]:
                referenced = {
                    token.split("{")[0].removesuffix("_bucket").removesuffix("_count")
                    for token in target["expr"].split()
                    if token.startswith("ml_platform_")
                }
                for name in referenced:
                    base = name.removesuffix("_total")
                    assert base in exported or name in exported, (panel["title"], name)
