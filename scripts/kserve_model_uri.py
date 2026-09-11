"""Derive the KServe storageUri for the promoted production model.

    python scripts/kserve_model_uri.py --tracking-uri http://localhost:5000

Prints the `pvc://` URI the InferenceService should carry, resolved from the
registry alias rather than from anyone remembering a run id.

Why this exists. The InferenceService needs a concrete path on the tracking
server's volume, and that path contains the MLflow run id of the promoted
version. Typing it by hand is how a cluster ends up serving a model nobody
chose. This reads the `production` alias -- the same mechanism M6 uses to decide
what production is, and the same one the API uses to resolve it -- and translates
the registry's own recorded source into the volume path underneath it:

    registry source   mlflow-artifacts:/1/<run id>/artifacts/model
    storageUri        pvc://<claim>/artifacts/1/<run id>/artifacts/model

It prints. It does not apply anything, and it does not watch for changes:
promotion moves the alias, and moving the cluster's model tier onto a new version
is a deliberate act, not a reconciliation loop.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ml_platform.config import load_config

#: What the tracking server records when it brokers its own artifacts.
ARTIFACT_SCHEME = "mlflow-artifacts:/"

#: Subdirectory of the volume the server is started with, from
#: `--artifacts-destination=/mlflow/artifacts` in k8s/base.
VOLUME_SUBPATH = "artifacts"


class ResolutionError(RuntimeError):
    """Raised when the registry cannot name a production model."""


def storage_uri(source: str, claim: str) -> str:
    """Translate a registry `source` into a KServe `pvc://` URI."""
    if not source.startswith(ARTIFACT_SCHEME):
        raise ResolutionError(
            f"the registered version's source is {source!r}, which is not served by the "
            "tracking server. A file:// source is a local path and is not reachable "
            "from a pod; see docs/kubernetes.md."
        )
    relative = source[len(ARTIFACT_SCHEME) :].lstrip("/")
    return f"pvc://{claim}/{VOLUME_SUBPATH}/{relative}"


def resolve(tracking_uri: str, claim: str, *, environment: str = "production") -> dict[str, Any]:
    """The production version, and everything needed to trace it."""
    import mlflow

    config = load_config(environment)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    name = config.registered_model_name
    alias = config.production_alias
    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise ResolutionError(
            f"no {alias!r} alias on {name!r} at {tracking_uri}. "
            "Seed the registry first: scripts/seed_registry.py"
        ) from exc

    tags = dict(version.tags or {})
    return {
        "registered_model": name,
        "alias": alias,
        "version": str(version.version),
        "mlflow_run_id": str(version.run_id),
        "platform_run_id": tags.get("platform_run_id", ""),
        "source": str(version.source),
        "storage_uri": storage_uri(str(version.source), claim),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the KServe storageUri for production.")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="the cluster's tracking server, usually via kubectl port-forward",
    )
    parser.add_argument("--claim", default="mlflow-store", help="PersistentVolumeClaim name")
    parser.add_argument("--environment", default="production")
    parser.add_argument(
        "--quiet", action="store_true", help="print only the URI, for use in a pipeline"
    )
    args = parser.parse_args(argv)

    try:
        resolved = resolve(args.tracking_uri, args.claim, environment=args.environment)
    except ResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.quiet:
        print(resolved["storage_uri"])
        return 0

    width = max(len(key) for key in resolved)
    for key, value in resolved.items():
        print(f"{key.replace('_', ' '):<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
