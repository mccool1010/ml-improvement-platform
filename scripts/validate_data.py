"""Validate the dataset. Thin wrapper over ``python -m ml_platform validate``."""

from __future__ import annotations

import sys

from ml_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
