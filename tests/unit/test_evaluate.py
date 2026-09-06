"""Tests for evaluation metrics.

These guard the two properties the promotion gates depend on: that the operating
point means what it says, and that a model with good ranking but bad calibration
is visibly distinguishable from a good one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_platform.models.evaluate import (
    brier_skill,
    evaluate_predictions,
    lift_over_base_rate,
    metrics_at_capacity,
)

RNG = np.random.default_rng(0)


def _separable(n: int = 2000, positive_rate: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores where the ranking is perfect."""
    y = (RNG.random(n) < positive_rate).astype(int)
    scores = np.where(y == 1, RNG.uniform(0.6, 1.0, n), RNG.uniform(0.0, 0.4, n))
    return y, scores


class TestMetricsAtCapacity:
    def test_perfect_ranking_recovers_all_positives_within_capacity(self) -> None:
        y = np.array([0] * 90 + [1] * 10)
        scores = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.6, 1.0, 10)])
        precision, recall, lift, _ = metrics_at_capacity(y, scores, 0.10)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert lift == pytest.approx(10.0)

    def test_random_scores_give_lift_near_one(self) -> None:
        y = np.array([0] * 9000 + [1] * 1000)
        scores = RNG.random(10000)
        _, _, lift, _ = metrics_at_capacity(y, scores, 0.10)
        assert 0.7 < lift < 1.3

    def test_capacity_outside_the_unit_interval_is_rejected(self) -> None:
        y, scores = _separable()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="capacity"):
                metrics_at_capacity(y, scores, bad)

    def test_larger_capacity_never_reduces_recall(self) -> None:
        y, scores = _separable()
        recalls = [metrics_at_capacity(y, scores, c)[1] for c in (0.05, 0.10, 0.25, 0.50)]
        assert recalls == sorted(recalls)


class TestBrierSkill:
    def test_constant_base_rate_prediction_scores_zero_skill(self) -> None:
        y = np.array([0] * 900 + [1] * 100)
        scores = np.full(1000, 0.1)
        from sklearn.metrics import brier_score_loss

        assert brier_skill(y, float(brier_score_loss(y, scores))) == pytest.approx(0.0, abs=1e-9)

    def test_a_worse_than_trivial_model_scores_negative(self) -> None:
        """This is the check that caught balanced class weights.

        Inflated probabilities left ranking intact but pushed the Brier score
        past that of a constant base-rate predictor.
        """
        y = np.array([0] * 900 + [1] * 100)
        inflated = np.where(y == 1, 0.9, 0.45)
        result = evaluate_predictions(y, inflated, split="t")
        assert result.brier_skill_score < 0


class TestEvaluatePredictions:
    def test_calibration_ratio_detects_systematic_under_prediction(self) -> None:
        y = np.array([0] * 800 + [1] * 200)
        under = np.where(y == 1, 0.2, 0.05)
        result = evaluate_predictions(y, under, split="t")
        assert result.calibration_ratio < 1.0

    def test_calibration_ratio_is_one_when_mean_matches_base_rate(self) -> None:
        y = np.array([0] * 900 + [1] * 100)
        scores = np.full(1000, 0.1)
        result = evaluate_predictions(y, scores, split="t")
        assert result.calibration_ratio == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_predictions(np.array([0, 1]), np.array([0.5]), split="t")

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            evaluate_predictions(np.array([]), np.array([]), split="t")

    def test_single_class_split_is_rejected(self) -> None:
        """Average precision and ROC AUC are undefined with one class."""
        with pytest.raises(ValueError, match="single class"):
            evaluate_predictions(np.zeros(10, dtype=int), RNG.random(10), split="t")

    def test_lift_over_base_rate_is_one_for_a_random_model(self) -> None:
        y = np.array([0] * 9000 + [1] * 1000)
        result = evaluate_predictions(y, RNG.random(10000), split="t")
        assert 0.8 < lift_over_base_rate(result) < 1.25
