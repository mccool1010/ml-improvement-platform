"""Download and verify the raw SBA register.

python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_platform.config import load_config
from ml_platform.data import ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and checksum the raw dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    config = load_config("production")
    path = ingestion.acquire(config.source, config.raw_path, force=args.force)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"verified {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
