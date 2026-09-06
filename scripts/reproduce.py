"""Verify reproducibility. Thin wrapper over ``python -m ml_platform reproduce``."""

from __future__ import annotations

import sys

from ml_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce", *sys.argv[1:]]))
