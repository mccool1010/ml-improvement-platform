"""Model fitting.

Kept deliberately thin. Orchestration, provenance and reporting live in
``ml_platform.pipelines.train_pipeline``; this module only turns a split plus a
model specification into a fitted pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_platform.data.splitting import Split
from ml_platform.features.engineering import build_features
from ml_platform.models.baseline import build_model

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted pipeline together with what it was trained on."""

    name: str
    feature_set: str
    pipeline: Pipeline
    n_train_rows: int
    train_seconds: float
    spec: dict[str, Any]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "feature_set": self.feature_set,
                "pipeline": self.pipeline,
                "spec": self.spec,
            },
            path,
        )
        return path


def load_model(path: Path) -> TrainedModel:
    """Load a model saved by :meth:`TrainedModel.save`."""
    payload = joblib.load(path)
    return TrainedModel(
        name=payload["name"],
        feature_set=payload["feature_set"],
        pipeline=payload["pipeline"],
        n_train_rows=-1,
        train_seconds=0.0,
        spec=payload["spec"],
    )


def features_and_target(
    split: Split, feature_set: str, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix and target vector for a split."""
    features = build_features(split.frame, feature_set)
    target = split.frame[target_column].astype(int)
    return features, target


def train(spec: dict[str, Any], split: Split, target_column: str) -> TrainedModel:
    """Fit the model described by ``spec`` on ``split``."""
    feature_set = str(spec["feature_set"])
    features, target = features_and_target(split, feature_set, target_column)

    if target.nunique() < 2:
        raise ValueError(f"split {split.name!r} contains a single class; cannot fit a classifier")

    pipeline = build_model(spec)
    LOGGER.info(
        "fitting %s on %s rows (%s features, positive rate %.4f)",
        spec["name"],
        len(features),
        features.shape[1],
        target.mean(),
    )
    started = time.perf_counter()
    pipeline.fit(features, target)
    elapsed = time.perf_counter() - started
    LOGGER.info("fitted %s in %.1fs", spec["name"], elapsed)

    return TrainedModel(
        name=str(spec["name"]),
        feature_set=feature_set,
        pipeline=pipeline,
        n_train_rows=len(features),
        train_seconds=round(elapsed, 3),
        spec=spec,
    )
