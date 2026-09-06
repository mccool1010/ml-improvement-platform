"""Download the register. Thin wrapper over ``python -m ml_platform download``."""

from __future__ import annotations

import sys

from ml_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
