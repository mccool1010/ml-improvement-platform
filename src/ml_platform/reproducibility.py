"""Run provenance.

A metric is only evidence if you can say what produced it. Every run records the
code version, the resolved configuration, the checksum of the input data, the
library versions, the determinism controls in force, and the interpreter. A
result that cannot be tied to those six is not reproducible, and the record says
so rather than implying otherwise.

Nothing here writes an absolute filesystem path into a record. Paths are stored
relative to the project root so that two machines produce comparable records.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_platform.determinism import DeterminismSettings, set_global_seed
from ml_platform.paths import project_root

#: Libraries whose version can move a metric.
TRACKED_LIBRARIES: tuple[str, ...] = ("pandas", "numpy", "sklearn", "scipy", "pandera", "joblib")

__all__ = [
    "RunContext",
    "git_is_dirty",
    "git_revision",
    "library_versions",
    "lockfile_digest",
    "new_run_context",
    "set_global_seed",
]


def _git(*args: str) -> str | None:
    """Run a git command inside the project root, or return None if unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=project_root(),
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def git_revision() -> str:
    """Current commit SHA, or ``unknown`` outside a repository."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Results from a dirty tree cannot be reproduced from the commit alone, so this
    is recorded rather than assumed away.
    """
    return bool(_git("status", "--porcelain"))


def library_versions() -> dict[str, str]:
    """Versions of the libraries that can change a metric."""
    versions: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def lockfile_digest() -> str:
    """SHA-256 of ``uv.lock``, identifying the exact resolved dependency set."""
    lockfile = project_root() / "uv.lock"
    if not lockfile.exists():
        return "absent"
    return hashlib.sha256(lockfile.read_bytes()).hexdigest()[:16]


def interpreter_description() -> str:
    """Python version and build, which can affect floating-point behaviour."""
    return f"{platform.python_implementation()} {platform.python_version()}"


@dataclass
class RunContext:
    """Everything needed to tie a result back to what produced it."""

    run_id: str
    started_at: str
    git_revision: str
    git_dirty: bool
    config_fingerprint: str
    environment: str
    seed: int
    data_sha256: str
    lockfile_sha256: str
    determinism: dict[str, Any]
    interpreter: str = field(default_factory=interpreter_description)
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    libraries: dict[str, str] = field(default_factory=library_versions)
    command: str = field(default_factory=lambda: " ".join(Path(a).name for a in sys.argv[:1]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_reproducible(self) -> bool:
        """Whether this run could be reproduced from its recorded provenance."""
        return (
            self.git_revision != "unknown"
            and not self.git_dirty
            and self.lockfile_sha256 != "absent"
        )


def new_run_context(
    *,
    config_fingerprint: str,
    environment: str,
    seed: int,
    data_sha256: str,
    determinism: DeterminismSettings,
    prefix: str = "run",
) -> RunContext:
    """Create a run context stamped with the current time and code version."""
    now = datetime.now(UTC)
    return RunContext(
        run_id=f"{prefix}-{now.strftime('%Y%m%dT%H%M%SZ')}",
        started_at=now.isoformat(),
        git_revision=git_revision(),
        git_dirty=git_is_dirty(),
        config_fingerprint=config_fingerprint,
        environment=environment,
        seed=seed,
        data_sha256=data_sha256,
        lockfile_sha256=lockfile_digest(),
        determinism=determinism.to_dict(),
    )
