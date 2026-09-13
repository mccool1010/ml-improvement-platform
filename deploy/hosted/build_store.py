"""Rebuild a registry inside the hosted image from the exported bundle.

Runs at image build time. Recreates each recorded run with its original tags,
parameters, metrics and timestamps, re-logs the production model into the run
that trained it, registers it and sets the production alias with the version's
recorded tags. The result is an ordinary SQLite MLflow store whose artifact
paths are valid inside the container.
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag


def main(bundle: Path, store: Path, artifacts: Path) -> None:
    import mlflow
    import mlflow.sklearn

    manifest = json.loads((bundle / "runs.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
    client = mlflow.MlflowClient()

    experiment_id = client.create_experiment(
        manifest["experiment_name"], artifact_location=artifacts.as_uri()
    )

    production_source = manifest["production_version"]["source_run_id"]
    new_ids: dict[str, str] = {}
    for record in manifest["runs"]:
        run = client.create_run(
            experiment_id,
            start_time=record["start_time"],
            tags=record["tags"],
            run_name=record["run_name"],
        )
        timestamp = record["end_time"] or record["start_time"]
        client.log_batch(
            run.info.run_id,
            metrics=[Metric(k, float(v), timestamp, 0) for k, v in record["metrics"].items()],
            params=[Param(k, str(v)) for k, v in record["params"].items()],
            tags=[RunTag(k, v) for k, v in record["tags"].items()],
        )
        new_ids[record["source_run_id"]] = run.info.run_id
        if record["source_run_id"] != production_source:
            client.set_terminated(run.info.run_id, record["status"], record["end_time"])

    run_id = new_ids[production_source]
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(bundle / "model.tar.gz") as tar:
            tar.extractall(tmp, filter="data")
        pipeline = mlflow.sklearn.load_model(str(Path(tmp) / "model"))

    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

    name = manifest["registered_model_name"]
    client.create_registered_model(name)
    version = client.create_model_version(
        name,
        source=info.model_uri,
        run_id=run_id,
        tags=manifest["production_version"]["tags"],
    )
    client.set_registered_model_alias(name, manifest["production_alias"], version.version)
    print(f"store: {len(new_ids)} runs, {name} v{version.version} @ {manifest['production_alias']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
