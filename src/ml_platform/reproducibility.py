"""Run provenance and determinism.

A metric is only evidence if you can say what produced it. Every run records the
code version, the resolved configuration, the checksum of the input data, the
library versions, and the seed. Reproducing a number means re-running with the
same four, and a mismatch is visible rather than mysterious.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every source of randomness the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_revision() -> str:
    """Current commit SHA, or ``unknown`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_is_dirty() -> bool:
    """Whether the working tree has uncommitted changes.

    Results produced from a dirty tree are not reproducible from the commit
    alone, so this is recorded rather than assumed away.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(result.stdout.strip())


def library_versions() -> dict[str, str]:
    """Versions of the libraries that can change a metric."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "sklearn", "pandera"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


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
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    libraries: dict[str, str] = field(default_factory=library_versions)
    command: str = field(default_factory=lambda: " ".join(sys.argv))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_run_context(
    *,
    config_fingerprint: str,
    environment: str,
    seed: int,
    data_sha256: str,
    prefix: str = "run",
) -> RunContext:
    """Create a run context stamped with the current time and code version."""
    now = datetime.now(UTC)
    run_id = f"{prefix}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    return RunContext(
        run_id=run_id,
        started_at=now.isoformat(),
        git_revision=git_revision(),
        git_dirty=git_is_dirty(),
        config_fingerprint=config_fingerprint,
        environment=environment,
        seed=seed,
        data_sha256=data_sha256,
    )
