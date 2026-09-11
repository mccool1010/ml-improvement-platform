"""Turning a loan application into a default probability.

This is the only implementation of that step in the project. It is called in two
places and they must not diverge:

* in process, by :class:`ml_platform.api.model.LoadedModel`, which is how the
  local development server and the plain container serve;
* inside the KServe predictor, which is how the cluster serves.

Two things have to happen together and in this order: the register-shaped frame
becomes features through the same code that built them for training, and the
pipeline produces the probability of the positive class. Splitting them across
tiers would let the served feature representation drift from the trained one,
which is the failure this arrangement exists to prevent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_platform.features.engineering import build_features

#: The feature set used when nothing says otherwise. The registered version
#: records its own in a `feature_set` tag, which is what the deployment passes.
DEFAULT_FEATURE_SET = "engineered"

#: Columns that must be real datetimes before features are built: the engineered
#: set takes a day difference between them and reads a month off one. Over HTTP
#: they travel as ISO strings, so they have to be coerced back explicitly. JSON
#: has no date type, and a column left as text fails on the `.dt` accessor.
DATE_COLUMNS: tuple[str, ...] = ("ApprovalDate", "DisbursementDate")


def frame_to_instances(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A register-shaped frame as JSON-safe records, for the wire."""
    prepared = frame.copy()
    for column in DATE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column]).dt.strftime("%Y-%m-%d")
    return [
        {str(key): (None if pd.isna(value) else value) for key, value in record.items()}
        for record in prepared.to_dict(orient="records")
    ]


def instances_to_frame(instances: list[dict[str, Any]]) -> pd.DataFrame:
    """The inverse of :func:`frame_to_instances`, on the receiving side.

    These two are a pair. If they ever disagree the served feature
    representation stops matching the trained one, silently, which is exactly
    the failure the single scoring path exists to prevent.
    """
    frame = pd.DataFrame(instances)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def score(
    pipeline: Any, frame: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET
) -> list[float]:
    """Probability of default for each row of a register-shaped frame.

    ``predict_proba``, not ``predict``. The project's decisions are made by
    comparing a probability against a review-capacity threshold, so a hard label
    from the model would discard the very number the threshold acts on.
    """
    features = build_features(frame, feature_set)
    return [float(p) for p in pipeline.predict_proba(features)[:, 1]]
