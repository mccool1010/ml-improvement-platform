"""Evaluation metrics.

Metric choice follows the decision the model actually informs: which loan
applications to send for manual review. The classes are imbalanced, so accuracy is
meaningless and ROC AUC is optimistic. Average precision is the primary metric.

**The operating point is a review capacity, not a fixed precision.** A reviewer
team can examine a roughly fixed share of applications per week, so the real
question is "of the riskiest 10% we can afford to look at, how many actual
charge-offs did we catch". A fixed precision target would also be unusable here:
the base rate moves from 6.8% to 33.8% across the study period, so any fixed
precision is trivially easy in one period and unreachable in another. Measured on
this data, the baseline cannot reach 50% precision at any threshold in the
training period, which would have made that gate permanently uninformative.

Calibration is reported separately and deliberately. Under the concept drift this
project demonstrates, a model can keep its ranking ability while its probabilities
become badly wrong. A gate watching only ranking would wave that through.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass
class EvaluationResult:
    """Metrics for one model on one dataset."""

    split: str
    n_rows: int
    n_positive: int
    positive_rate: float
    average_precision: float
    roc_auc: float
    brier_score: float
    brier_skill_score: float
    review_capacity: float
    precision_at_capacity: float
    recall_at_capacity: float
    lift_at_capacity: float
    threshold_at_capacity: float
    mean_predicted_probability: float
    calibration_ratio: float
    inference_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metrics_at_capacity(
    y_true: np.ndarray, y_score: np.ndarray, capacity: float
) -> tuple[float, float, float, float]:
    """Precision, recall, lift and threshold when flagging the top ``capacity`` share.

    ``capacity`` is the fraction of applications sent for review, so a capacity of
    0.10 scores the 10% of applications the model ranks riskiest.
    """
    if not 0.0 < capacity < 1.0:
        raise ValueError(f"review capacity must be in (0, 1), got {capacity}")

    threshold = float(np.quantile(y_score, 1.0 - capacity))
    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())
    total_positive = int(y_true.sum())

    if n_flagged == 0 or total_positive == 0:
        return 0.0, 0.0, 0.0, threshold

    precision = float(y_true[flagged].mean())
    recall = float(y_true[flagged].sum() / total_positive)
    base_rate = float(y_true.mean())
    lift = precision / base_rate if base_rate > 0 else float("nan")
    return precision, recall, lift, threshold


def brier_skill(y_true: np.ndarray, brier: float) -> float:
    """Brier score relative to always predicting the base rate.

    Positive means better than the trivial predictor, zero means no better, and
    negative means worse. The baseline's original balanced-class-weight
    configuration scored negative here, which is what flagged it as unusable.
    """
    base_rate = float(y_true.mean())
    reference = base_rate * (1.0 - base_rate)
    if reference <= 0:
        return float("nan")
    return 1.0 - (brier / reference)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    split: str,
    review_capacity: float = 0.10,
    inference_seconds: float = 0.0,
) -> EvaluationResult:
    """Compute the full metric set for one split."""
    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError(f"shape mismatch: y_true {truth.shape} vs y_score {scores.shape}")
    if truth.size == 0:
        raise ValueError("cannot evaluate an empty split")
    if truth.max() == truth.min():
        raise ValueError(f"split {split!r} has a single class; metrics are undefined")

    base_rate = float(truth.mean())
    mean_prediction = float(scores.mean())
    brier = float(brier_score_loss(truth, scores))

    precision, recall, lift, threshold = metrics_at_capacity(truth, scores, review_capacity)

    # Above 1.0 the model over-predicts risk, below 1.0 it under-predicts.
    calibration = mean_prediction / base_rate if base_rate > 0 else float("nan")

    return EvaluationResult(
        split=split,
        n_rows=int(truth.size),
        n_positive=int(truth.sum()),
        positive_rate=round(base_rate, 6),
        average_precision=round(float(average_precision_score(truth, scores)), 6),
        roc_auc=round(float(roc_auc_score(truth, scores)), 6),
        brier_score=round(brier, 6),
        brier_skill_score=round(brier_skill(truth, brier), 6),
        review_capacity=review_capacity,
        precision_at_capacity=round(precision, 6),
        recall_at_capacity=round(recall, 6),
        lift_at_capacity=round(lift, 6),
        threshold_at_capacity=round(threshold, 6),
        mean_predicted_probability=round(mean_prediction, 6),
        calibration_ratio=round(float(calibration), 6),
        inference_seconds=round(inference_seconds, 4),
    )


def evaluate_model(
    model: Any,
    features: pd.DataFrame,
    y_true: pd.Series,
    *,
    split: str,
    review_capacity: float = 0.10,
) -> EvaluationResult:
    """Score ``features`` with ``model`` and evaluate, timing the inference pass."""
    started = time.perf_counter()
    scores = model.predict_proba(features)[:, 1]
    elapsed = time.perf_counter() - started
    return evaluate_predictions(
        y_true,
        scores,
        split=split,
        review_capacity=review_capacity,
        inference_seconds=elapsed,
    )


def lift_over_base_rate(result: EvaluationResult) -> float:
    """Average precision relative to a random model, which scores the base rate.

    A value of 1.0 means the model is worthless. This makes results comparable
    across splits whose base rates differ, which matters here because the base
    rate moves from 6.8% to 33.8% across the study period.
    """
    if result.positive_rate <= 0:
        return float("nan")
    return round(result.average_precision / result.positive_rate, 6)
