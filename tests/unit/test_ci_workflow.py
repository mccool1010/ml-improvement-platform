"""Tests for the CI workflow definition.

Same reasoning as :mod:`tests.unit.test_container`: the workflow is a
configuration file, and the mistakes worth catching in it are visible without a
runner. A workflow that has quietly stopped running mypy, or that marks a gate
``continue-on-error``, still shows a green tick. Nothing else in the project
would notice.

These tests read the YAML. They do not execute the workflow.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from ml_platform.paths import project_root

WORKFLOW = project_root() / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step of every job, flattened."""
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


@pytest.fixture(scope="module")
def commands(steps: list[dict[str, Any]]) -> str:
    """Every shell command in the workflow, concatenated."""
    return "\n".join(str(step.get("run", "")) for step in steps)


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block.

    YAML 1.1 reads a bare ``on`` as the boolean ``True``, so the key is looked up
    under both spellings rather than assuming which one the parser produced.
    """
    block = workflow.get("on", workflow.get(True))
    assert isinstance(block, dict), "the workflow has no trigger block"
    return block


class TestTheWorkflowIsValid:
    def test_the_workflow_exists(self) -> None:
        assert WORKFLOW.exists()

    def test_it_parses_as_yaml(self, workflow: dict[str, Any]) -> None:
        assert workflow["name"]
        assert workflow["jobs"]

    def test_every_job_has_steps(self, workflow: dict[str, Any]) -> None:
        for name, job in workflow["jobs"].items():
            assert job.get("steps"), f"job {name} has no steps"

    def test_every_job_checks_out_the_repository(self, workflow: dict[str, Any]) -> None:
        for name, job in workflow["jobs"].items():
            uses = [str(step.get("uses", "")) for step in job["steps"]]
            assert any(u.startswith("actions/checkout") for u in uses), f"job {name}"


class TestItRunsOnTheRightEvents:
    def test_it_runs_on_pushes_to_main(self, workflow: dict[str, Any]) -> None:
        assert "main" in _triggers(workflow)["push"]["branches"]

    def test_it_runs_on_pull_requests(self, workflow: dict[str, Any]) -> None:
        assert "pull_request" in _triggers(workflow)


class TestInstallationIsReproducible:
    def test_dependencies_come_from_the_lockfile(self, commands: str) -> None:
        """`--frozen` is what ties CI to the recorded dependency set."""
        assert "uv sync --frozen" in commands

    def test_no_job_installs_without_the_lockfile(self, commands: str) -> None:
        for line in commands.splitlines():
            if "uv sync" in line:
                assert "--frozen" in line, f"unpinned install: {line.strip()}"

    def test_the_dev_extra_is_requested(self, commands: str) -> None:
        """pytest, ruff and mypy live in the dev extra, not the base dependencies."""
        assert "--extra dev" in commands

    def test_uv_is_pinned_to_a_version(self, steps: list[dict[str, Any]]) -> None:
        setups = [s for s in steps if str(s.get("uses", "")).startswith("astral-sh/setup-uv")]
        assert setups, "the workflow does not install uv"
        for step in setups:
            assert step.get("with", {}).get("version"), "uv version is not pinned"

    def test_the_runner_image_is_pinned(self, workflow: dict[str, Any]) -> None:
        """`ubuntu-latest` changing underneath the project would look like a regression."""
        for name, job in workflow["jobs"].items():
            assert job["runs-on"] != "ubuntu-latest", f"job {name} floats with the runner image"


class TestEveryCheckIsRun:
    @pytest.mark.parametrize(
        "command",
        [
            "ruff check",
            "ruff format --check",
            "mypy",
            "pytest",
            "ml_platform reproduce",
        ],
    )
    def test_the_check_is_present(self, commands: str, command: str) -> None:
        assert command in commands

    def test_the_image_is_built(self, steps: list[dict[str, Any]]) -> None:
        uses = [str(step.get("uses", "")) for step in steps]
        assert any(u.startswith("docker/build-push-action") for u in uses)


class TestNoGateIsBypassed:
    """A green tick has to mean the checks ran and passed."""

    def test_no_job_continues_on_error(self, workflow: dict[str, Any]) -> None:
        for name, job in workflow["jobs"].items():
            assert not job.get("continue-on-error"), f"job {name} cannot fail the workflow"

    def test_no_check_step_continues_on_error(self, steps: list[dict[str, Any]]) -> None:
        for step in steps:
            # Cleanup steps legitimately run with `if: always()`; none of them
            # may also swallow their own failure.
            assert not step.get("continue-on-error"), f"step {step.get('name')} cannot fail"

    def test_pytest_is_not_narrowed(self, commands: str) -> None:
        """Deselecting tests in CI would make the suite mean less than it does locally."""
        for line in commands.splitlines():
            if "pytest" in line:
                assert "--deselect" not in line
                assert "-k " not in line
                assert "--ignore" not in line

    def test_the_reproduction_uses_a_defined_tolerance_profile(self, commands: str) -> None:
        """CI may pick a profile; it may not invent tolerances of its own."""
        from ml_platform.pipelines.reproduce import load_reference

        profiles = set(load_reference()["tolerances"])
        used = [
            line.split("--profile", 1)[1].split()[0]
            for line in commands.splitlines()
            if "--profile" in line
        ]
        assert used, "the reproduction step names no profile"
        for profile in used:
            assert profile in profiles, f"{profile} is not defined in configs/reference.yaml"

    def test_the_docker_build_failure_is_not_swallowed(self, commands: str) -> None:
        """`|| true` belongs on cleanup, never on anything that verifies something."""
        for line in commands.splitlines():
            if "|| true" in line:
                assert "docker logs" in line or "docker rm" in line, line.strip()


class TestItDependsOnNothingLocal:
    """CI must not reach for a developer's machine."""

    def test_no_windows_paths_appear(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "C:/" not in text
        assert "C:\\" not in text

    def test_the_local_mlflow_store_is_not_mounted(self, commands: str) -> None:
        """Its artifact URIs are absolute host paths; see docs/ci.md."""
        assert "mlflow.db" not in commands
        assert "mlartifacts" not in commands

    def test_the_dataset_cache_is_keyed_on_the_recorded_checksum(
        self, steps: list[dict[str, Any]]
    ) -> None:
        """A mutable key could serve bytes the reference was never produced from."""
        caches = [s for s in steps if str(s.get("uses", "")).startswith("actions/cache")]
        assert caches, "the register is downloaded on every run"
        for step in caches:
            assert "dataset.outputs.sha" in str(step["with"]["key"])


class TestNoSecrets:
    def test_the_workflow_requests_no_secret(self) -> None:
        assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")

    def test_the_image_is_not_pushed(self, steps: list[dict[str, Any]]) -> None:
        for step in steps:
            if str(step.get("uses", "")).startswith("docker/build-push-action"):
                assert step["with"]["push"] is False

    def test_permissions_are_read_only(self, workflow: dict[str, Any]) -> None:
        assert workflow["permissions"] == {"contents": "read"}
