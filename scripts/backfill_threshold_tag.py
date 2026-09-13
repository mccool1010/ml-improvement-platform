"""Backfill the serving threshold tag onto a registered version that predates it.

The API serves at ``validation_threshold_at_capacity``, read from the model
version's tags. Registration writes that tag now, but versions registered before
it was added carry none, so the API falls back to a neutral 0.5 and says
``threshold_source: fallback``. For this model that flags almost nothing: the
validated operating point is about 0.065.

This copies the value the version's own training run already measured onto the
version. It invents nothing: if the run did not record the metric, it refuses.

    uv run python scripts/backfill_threshold_tag.py                       # local store
    uv run python scripts/backfill_threshold_tag.py --tracking-uri http://localhost:5000
    uv run python scripts/backfill_threshold_tag.py --dry-run

    # a seeded cluster registry: its copied run has no metrics, so read the
    # value from the original run in the store it was seeded from
    uv run python scripts/backfill_threshold_tag.py --tracking-uri http://localhost:5000 --source-tracking-uri sqlite:///mlflow.db

The API loads its model once at startup, so restart it afterwards
(``kubectl -n ml-platform rollout restart deploy/inference-api`` in the cluster).
"""

from __future__ import annotations

import argparse
import sys

TAG = "validation_threshold_at_capacity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", help="MLflow server; defaults to the configured store")
    parser.add_argument("--alias", default=None, help="alias to backfill (default: production)")
    parser.add_argument(
        "--source-tracking-uri",
        help="store holding the original training run, for a version seeded from elsewhere",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = parser.parse_args()

    from ml_platform.config import load_config, override_tracking_uri

    if args.tracking_uri:
        override_tracking_uri(args.tracking_uri)

    import mlflow

    config = load_config("production")
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()
    name = config.registered_model_name
    alias = args.alias or config.production_alias

    version = client.get_model_version_by_alias(name, alias)
    tags = dict(version.tags or {})
    print(f"{name} v{version.version} @ {alias}  (run {version.run_id})")

    if TAG in tags:
        print(f"already tagged: {TAG} = {tags[TAG]}; nothing to do")
        return 0

    metrics = client.get_run(version.run_id).data.metrics
    if TAG in metrics:
        value = metrics[TAG]
        print(f"run measured {TAG} = {value:.6f}")
    elif args.source_tracking_uri:
        value_or_none = _from_source(args.source_tracking_uri, tags)
        if value_or_none is None:
            return 1
        value = value_or_none
    else:
        print(
            f"refusing: run {version.run_id} recorded no {TAG} metric "
            "(pass --source-tracking-uri if this version was seeded from another store)",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        print("dry run: no tag written")
        return 0

    client.set_model_version_tag(name, version.version, TAG, f"{value:.6f}")
    print("tag written; restart the API so it reloads the version")
    return 0


def _from_source(source_uri: str, tags: dict[str, str]) -> float | None:
    """Read the metric from the original run, proven to be the same model.

    Seeding copies version tags verbatim, including the original ``mlflow_run_id``
    and ``platform_run_id``. The source run must carry the same platform run id,
    or it is not this model and nothing is written.
    """
    import mlflow

    original_run = tags.get("mlflow_run_id")
    platform_run = tags.get("platform_run_id")
    if not original_run or not platform_run:
        print(
            "refusing: version carries no mlflow_run_id/platform_run_id provenance", file=sys.stderr
        )
        return None

    source = mlflow.MlflowClient(tracking_uri=source_uri)
    run = source.get_run(original_run)
    if run.data.tags.get("platform_run_id") != platform_run:
        print(
            f"refusing: source run {original_run} is platform run "
            f"{run.data.tags.get('platform_run_id')!r}, not {platform_run!r}",
            file=sys.stderr,
        )
        return None
    if TAG not in run.data.metrics:
        print(f"refusing: source run {original_run} recorded no {TAG} metric", file=sys.stderr)
        return None
    value = float(run.data.metrics[TAG])
    print(f"source run {original_run} ({platform_run}) measured {TAG} = {value:.6f}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
