"""Pack the production model and the recorded run history for the hosted demo.

The hosted instance has no MLflow server to talk to, so it carries a small,
self-contained registry built from this bundle (see ``build_store.py``). The
bundle holds exactly what the local store recorded -- run tags, parameters,
metrics and timestamps, plus the production version's model artifact and tags --
so the hosted dashboard shows real history rather than a demo fixture.

    uv run python deploy/hosted/export_bundle.py

Writes ``deploy/hosted/bundle/`` (git-ignored): ``runs.json`` and ``model.tar.gz``.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"


def main() -> None:
    import mlflow

    from ml_platform.config import load_config

    config = load_config("production")
    mlflow.set_tracking_uri(config.tracking_uri)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise SystemExit(f"no experiment {config.experiment_name!r} in {config.tracking_uri}")

    version = client.get_model_version_by_alias(
        config.registered_model_name, config.production_alias
    )

    runs = []
    for run in client.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time ASC"], max_results=1000
    ):
        runs.append(
            {
                "source_run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "tags": {
                    k: v
                    for k, v in run.data.tags.items()
                    if not k.startswith("mlflow.") or k == "mlflow.runName"
                },
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }
        )

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    manifest = {
        "experiment_name": config.experiment_name,
        "registered_model_name": config.registered_model_name,
        "production_alias": config.production_alias,
        "production_version": {
            "source_run_id": version.run_id,
            "tags": dict(version.tags or {}),
        },
        "runs": runs,
    }
    (BUNDLE / "runs.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tmp)
        with tarfile.open(BUNDLE / "model.tar.gz", "w:gz") as tar:
            tar.add(local, arcname="model")

    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    print(f"bundle: {len(runs)} runs, production v{version.version}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
