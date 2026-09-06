"""Project paths.

Every path in the project resolves from the location of this file, never from the
current working directory and never from an absolute path baked into a config.
That means ``python scripts/train.py`` and ``python ../mlops/scripts/train.py``
write to the same place, and a checkout on another machine behaves identically.

The one exception is an explicit override through ``ML_PLATFORM_ROOT``, which
exists so an out-of-tree run (a container, a CI workspace with a read-only
checkout) can redirect the writable directories without editing configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root, three levels up from ``src/ml_platform/paths.py``.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def project_root() -> Path:
    """Repository root, overridable with ``ML_PLATFORM_ROOT``."""
    override = os.environ.get("ML_PLATFORM_ROOT")
    return Path(override).resolve() if override else PROJECT_ROOT


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the project root.

    Absolute inputs are returned unchanged, so a configuration may point at data
    outside the repository when it genuinely needs to.
    """
    if not parts:
        return project_root()
    first = Path(parts[0])
    if first.is_absolute():
        return Path(*parts)
    return project_root().joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_root(path: Path) -> str:
    """Render a path relative to the project root for logs and run records.

    Absolute machine-specific paths must never reach a committed artifact, or
    the record stops being comparable across machines.
    """
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


CONFIG_DIR = PROJECT_ROOT / "configs"
