"""Tests for the container configuration.

These read the Dockerfile and `.dockerignore` as text. That is a deliberate
choice: building an image in a unit test would need a daemon and minutes per
run, while the mistakes worth catching here are all visible in the files.

The properties under test are the ones that would be expensive to discover
later: a 179 MB dataset baked into the image, a registry state frozen at build
time, the service running as root, or the API being launched by something other
than the real application.
"""

from __future__ import annotations

import re

import pytest

from ml_platform.paths import project_root

DOCKERFILE = project_root() / "docker" / "Dockerfile"
DOCKERIGNORE = project_root() / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class TestFilesExist:
    def test_the_dockerfile_is_present(self) -> None:
        assert DOCKERFILE.exists()

    def test_the_dockerignore_is_present(self) -> None:
        """Without it the build context includes the 179 MB register."""
        assert DOCKERIGNORE.exists()


class TestBuildIsReproducible:
    def test_dependencies_install_from_the_lockfile(self, dockerfile: str) -> None:
        """`--frozen` is what ties the image to the recorded dependency set."""
        assert "uv sync --frozen" in dockerfile

    def test_the_lockfile_is_copied(self, dockerfile: str) -> None:
        assert re.search(r"COPY .*uv\.lock", dockerfile)

    def test_development_dependencies_are_excluded(self, dockerfile: str) -> None:
        assert "--no-dev" in dockerfile

    def test_the_python_version_matches_the_project(self, dockerfile: str) -> None:
        """The project pins >=3.12,<3.13; a different runtime breaks the contract."""
        assert "python:3.12" in dockerfile

    def test_the_base_image_is_pinned_to_a_distribution(self, dockerfile: str) -> None:
        assert "slim-bookworm" in dockerfile


class TestRuntimeSafety:
    def test_the_service_does_not_run_as_root(self, dockerfile: str) -> None:
        assert re.search(r"^USER\s+appuser", dockerfile, re.MULTILINE)

    def test_a_non_root_user_is_created(self, dockerfile: str) -> None:
        assert "useradd" in dockerfile

    def test_the_user_switch_precedes_the_command(self, dockerfile: str) -> None:
        """A USER after CMD would leave the process running as root."""
        assert dockerfile.index("USER appuser") < dockerfile.index("CMD [")

    def test_native_thread_pools_are_pinned(self, dockerfile: str) -> None:
        """uvicorn imports the app directly and never passes through the CLI
        bootstrap, so the determinism contract has to be set in the environment."""
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert f"{variable}=1" in dockerfile


class TestServiceContract:
    def test_the_api_port_is_exposed(self, dockerfile: str) -> None:
        assert re.search(r"^EXPOSE\s+8000", dockerfile, re.MULTILINE)

    def test_the_command_runs_the_existing_application(self, dockerfile: str) -> None:
        """There must be one serving implementation, not a container-only copy."""
        assert "ml_platform.api.main:app" in dockerfile

    def test_the_server_listens_on_all_interfaces(self, dockerfile: str) -> None:
        """Binding to localhost inside a container makes it unreachable."""
        assert "--host" in dockerfile
        assert "0.0.0.0" in dockerfile

    def test_a_healthcheck_is_configured(self, dockerfile: str) -> None:
        assert "HEALTHCHECK" in dockerfile

    def test_the_healthcheck_uses_liveness_not_readiness(self, dockerfile: str) -> None:
        """A container with no promoted model is alive and should not be killed;
        withholding traffic is what /ready is for."""
        healthcheck = dockerfile[dockerfile.index("HEALTHCHECK") :]
        assert "/health" in healthcheck
        assert "/ready" not in healthcheck

    def test_configuration_is_present_in_the_image(self, dockerfile: str) -> None:
        """The app reads configs/ at startup; without it nothing resolves."""
        assert re.search(r"COPY .*configs/", dockerfile)

    def test_the_source_is_present_in_the_image(self, dockerfile: str) -> None:
        assert re.search(r"COPY .*src/", dockerfile)


class TestBuildContextExcludesTheExpensiveThings:
    @pytest.mark.parametrize("entry", ["data/", ".git/", ".venv/", "tests/", "notebooks/"])
    def test_bulk_and_irrelevant_paths_are_excluded(
        self, dockerignore: list[str], entry: str
    ) -> None:
        assert entry in dockerignore

    @pytest.mark.parametrize("entry", ["mlruns/", "mlartifacts/", "mlflow.db", "models/"])
    def test_the_model_store_is_excluded(self, dockerignore: list[str], entry: str) -> None:
        """Baking the store in would freeze one registry state into the image, so
        promoting a model would require rebuilding it."""
        assert entry in dockerignore

    @pytest.mark.parametrize("entry", [".env", "*.joblib", "*.pkl"])
    def test_local_state_and_secrets_are_excluded(
        self, dockerignore: list[str], entry: str
    ) -> None:
        assert entry in dockerignore

    def test_the_lockfile_is_not_excluded(self, dockerignore: list[str]) -> None:
        """Excluding it would break the frozen install the build depends on."""
        assert "uv.lock" not in dockerignore

    def test_configuration_is_not_excluded(self, dockerignore: list[str]) -> None:
        assert not any(entry.rstrip("/") == "configs" for entry in dockerignore)

    def test_source_is_not_excluded(self, dockerignore: list[str]) -> None:
        assert not any(entry.rstrip("/") == "src" for entry in dockerignore)
