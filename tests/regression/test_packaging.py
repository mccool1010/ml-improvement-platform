"""The repository must be complete enough to build from a clean clone.

This exists because it was not. The ``.gitignore`` pattern ``models/`` was
intended to exclude trained model binaries at the repository root, but an
unanchored pattern matches at every level, so it also excluded
``src/ml_platform/models/``. The working tree was fine and every test passed. A
fresh clone could not import ``ml_platform.models`` at all.

Nothing in a normal test run catches that, because a normal test run uses the
working tree. These tests inspect what git actually tracks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ml_platform.paths import project_root

SOURCE_DIRS = ("src", "tests", "scripts", "configs")


def _tracked_files() -> set[Path]:
    """Every path git tracks, relative to the project root."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        cwd=project_root(),
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def _is_ignored(relative: Path) -> bool:
    """Whether git would ignore this path."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(relative)],
        capture_output=True,
        timeout=30,
        cwd=project_root(),
    )
    return result.returncode == 0


@pytest.fixture(scope="module")
def tracked() -> set[Path]:
    return _tracked_files()


class TestSourceIsTracked:
    def test_git_is_available(self, tracked: set[Path]) -> None:
        """Guard: an empty listing would make every other test pass vacuously."""
        assert tracked, "git ls-files returned nothing"

    @pytest.mark.parametrize("directory", SOURCE_DIRS)
    def test_every_python_file_is_tracked(self, directory: str, tracked: set[Path]) -> None:
        root = project_root()
        untracked = []
        for path in (root / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            if Path(relative.as_posix()) not in tracked:
                untracked.append(relative.as_posix())
        assert not untracked, f"source files not tracked by git: {untracked}"

    def test_every_package_directory_is_importable(self, tracked: set[Path]) -> None:
        """Each package under src needs a tracked ``__init__.py``."""
        root = project_root()
        package_root = root / "src" / "ml_platform"
        missing = []
        for directory in [package_root, *(d for d in package_root.rglob("*") if d.is_dir())]:
            if "__pycache__" in directory.parts:
                continue
            init = (directory / "__init__.py").relative_to(root)
            if Path(init.as_posix()) not in tracked:
                missing.append(init.as_posix())
        assert not missing, f"package directories without a tracked __init__.py: {missing}"

    def test_configuration_files_are_tracked(self, tracked: set[Path]) -> None:
        root = project_root()
        untracked = [
            path.relative_to(root).as_posix()
            for path in (root / "configs").glob("*.yaml")
            if Path(path.relative_to(root).as_posix()) not in tracked
        ]
        assert not untracked, f"configuration files not tracked: {untracked}"


class TestIgnorePatternsAreAnchored:
    """Ignore rules must not reach into the source tree."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/ml_platform/models/__init__.py",
            "src/ml_platform/models/train.py",
            "src/ml_platform/models/baseline.py",
            "src/ml_platform/models/evaluate.py",
            "src/ml_platform/data/validation.py",
            "src/ml_platform/pipelines/reproduce.py",
        ],
    )
    def test_source_paths_are_not_ignored(self, path: str) -> None:
        assert not _is_ignored(Path(path)), f"{path} is excluded by .gitignore"

    @pytest.mark.parametrize("path", ["models/x.joblib", "mlruns/0/meta.yaml", "data/raw/big.csv"])
    def test_generated_paths_are_still_ignored(self, path: str) -> None:
        """The original intent must survive the fix."""
        assert _is_ignored(Path(path)), f"{path} should be ignored but is not"


class TestReferenceArtifactsAreTracked:
    def test_the_locked_reference_is_tracked(self, tracked: set[Path]) -> None:
        """Without it, a fresh clone cannot verify reproduction."""
        assert Path("configs/reference.yaml") in tracked

    def test_the_lockfile_is_tracked(self, tracked: set[Path]) -> None:
        assert Path("uv.lock") in tracked
