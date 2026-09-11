"""Copy the promoted production model into another MLflow registry.

    python scripts/seed_registry.py --destination http://localhost:5000

This exists because of one fact established in M8: the local file-backed store
records each experiment's artifact root as an absolute host URI, so its rows are
not portable. A cluster therefore needs its own tracking server, and that
server starts empty. This moves the already-promoted model into it.

**This is a transfer, not a promotion.** It does not evaluate a gate, compare a
candidate, or decide anything. It refuses to run unless the source registry
already carries the production alias, and it copies the version tags verbatim,
so the registered version in the destination still records the platform run id,
the git revision, and the gate counts of the decision that M6 actually made.
Nothing here can turn a rejected model into a production one.

The artifact is re-logged rather than re-serialised: the source model is
downloaded and the same files are logged to the destination, so the bytes the
gates were evaluated against are the bytes that get served.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from ml_platform.config import ENV_TRACKING_URI, load_config
from ml_platform.determinism import pin_threads

LOGGER = logging.getLogger("seed_registry")


def _client(tracking_uri: str) -> Any:
    import mlflow

    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def _source_version(client: Any, name: str, alias: str) -> Any:
    """The version carrying the production alias, or a hard failure."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise SystemExit(
            f"the source registry has no {alias!r} alias on {name!r}: {exc}\n"
            "There is nothing to copy. Promote a candidate first."
        ) from exc


def _configured_source() -> str:
    """The project's own store, with MLFLOW_TRACKING_URI ignored.

    That variable points at the *destination* while seeding, so reading the
    source through it would ask the empty registry to copy itself.
    """
    previous = os.environ.pop(ENV_TRACKING_URI, None)
    try:
        return load_config("production").tracking_uri
    finally:
        if previous is not None:
            os.environ[ENV_TRACKING_URI] = previous


def seed(
    destination: str,
    *,
    environment: str = "production",
    source: str | None = None,
    dry_run: bool = False,
) -> int:
    import mlflow

    config = load_config(environment)
    pin_threads(config.n_threads)

    name = config.registered_model_name
    alias = config.production_alias

    source_uri = source or _configured_source()
    if source_uri == destination:
        raise SystemExit("source and destination are the same registry; nothing to do")

    source_client = _client(source_uri)
    version = _source_version(source_client, name, alias)
    tags = dict(version.tags or {})

    LOGGER.info("source      %s", source_uri)
    LOGGER.info("destination %s", destination)
    LOGGER.info("copying     %s v%s (run %s)", name, version.version, version.run_id)
    LOGGER.info("tags        %d carried over", len(tags))

    if dry_run:
        LOGGER.info("dry run; nothing written")
        return 0

    with tempfile.TemporaryDirectory() as scratch:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=version.source, dst_path=scratch, tracking_uri=source_uri
        )
        LOGGER.info("downloaded  %s", Path(local).name)

        mlflow.set_tracking_uri(destination)
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run(run_name=f"seeded-{name}-v{version.version}") as run:
            # Recorded so the copy can always be told from a training run, and
            # traced back to the run that actually produced the model.
            mlflow.set_tags(
                {
                    **tags,
                    "seeded_from_run_id": str(version.run_id),
                    "seeded_from_version": str(version.version),
                    "seeded": "true",
                }
            )
            mlflow.log_artifacts(local, artifact_path="model")
            target_run = run.info.run_id

        registered = mlflow.register_model(
            model_uri=f"runs:/{target_run}/model", name=name, tags=tags
        )
        _client(destination).set_registered_model_alias(name, alias, registered.version)

    LOGGER.info("registered  %s v%s and set alias %r", name, registered.version, alias)
    print(f"{name} v{registered.version} now carries the {alias!r} alias on {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        required=True,
        help="tracking URI to copy into, e.g. http://localhost:5000",
    )
    parser.add_argument("--environment", default="production")
    parser.add_argument(
        "--source", default=None, help="tracking URI to copy from (default: the project's store)"
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be copied")
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)-7s %(name)s | %(message)s")
    # MLflow prints a run URL with an emoji in it. The default Windows console
    # encoding cannot represent that and raises mid-run, after the artifacts have
    # been logged but before the model is registered.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    return seed(
        args.destination,
        environment=args.environment,
        source=args.source,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
