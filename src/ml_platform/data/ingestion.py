"""Dataset acquisition.

The raw file is not committed. It is downloaded on demand and verified against a
recorded SHA-256, so every run either uses the exact bytes the recorded results
were produced from, or fails. A silently changed upstream mirror would otherwise
be indistinguishable from a modelling regression.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
_CHUNK = 1 << 20


class DataIntegrityError(RuntimeError):
    """Raised when the downloaded file does not match its recorded checksum."""


def sha256(path: Path) -> str:
    """Stream a SHA-256 of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, *, force: bool = False) -> Path:
    """Fetch ``url`` to ``destination`` unless it is already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        LOGGER.info("using cached %s", destination)
        return destination

    LOGGER.info("downloading %s -> %s", url, destination)
    staging = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, staging.open("wb") as handle:
        while chunk := response.read(_CHUNK):
            handle.write(chunk)
    staging.replace(destination)
    return destination


def verify(path: Path, expected_sha256: str) -> None:
    """Raise unless ``path`` hashes to ``expected_sha256``."""
    actual = sha256(path)
    if actual != expected_sha256:
        raise DataIntegrityError(
            f"checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )


def acquire(source: dict[str, Any], raw_path: Path, *, force: bool = False) -> Path:
    """Download if needed, then verify. Returns the verified path."""
    download(str(source["url"]), raw_path, force=force)
    verify(raw_path, str(source["sha256"]))
    return raw_path


def load_raw(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read the register as strings, so parsing happens in one explicit place."""
    frame = pd.read_csv(path, dtype=str, nrows=nrows, low_memory=False)
    LOGGER.info("loaded %s rows from %s", len(frame), path.name)
    return frame
