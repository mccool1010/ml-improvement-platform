"""Input drift detection.

Compares the feature distributions a model was trained on against the
distributions arriving now, and says whether they have moved far enough to be
worth acting on.

**This measures inputs only, and it is not a measure of whether the model is
still any good.** Those are different signals on different clocks, and the
project keeps them apart deliberately (see docs/model_lifecycle.md). Input drift
is observable the moment data arrives and may trigger a retraining *attempt*;
realised performance needs matured labels, which on this dataset take five years,
and is the only thing that can say a model got worse. Nothing here produces a
label, estimates an accuracy, or rejects a model. A drift decision is a
suggestion to go and look, and the existing M6 gates remain the only thing that
can change what production is.

Two statistics per feature, because each catches what the other misses.

*Population Stability Index* compares binned mass and is the industry's usual
threshold-able number. It is stable on large samples and interpretable, but it
depends on the binning and is blind to a shift that moves mass within a bin.

*Kolmogorov-Smirnov* (numeric) and *chi-square* (categorical) are distribution
tests that need no binning choice for the numeric case. Their weakness is the
opposite one: on hundreds of thousands of rows they reject almost any difference,
because a large enough sample makes a trivial shift significant. That is why the
decision below is driven by PSI, an effect size, and the test statistic is
reported alongside as corroboration rather than used as the trigger.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ml_platform.features.engineering import (
    add_engineered_features,
    feature_columns,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml_platform.config import Config

LOGGER = logging.getLogger(__name__)

#: Conventional PSI reading, and the reason the defaults in configs/base.yaml are
#: what they are: below 0.1 nothing has meaningfully moved, 0.1-0.25 is a
#: noticeable shift worth watching, above 0.25 is a population that no longer
#: looks like the one the model learned.
PSI_MINOR = 0.1
PSI_MAJOR = 0.25

#: Mass given to a bin that is empty in one of the two samples. PSI divides by
#: the reference proportion, so an empty bin would otherwise be infinite and one
#: unseen category would dominate every other signal.
EPSILON = 1e-6

#: Numeric bin count. Ten quantile bins is the usual choice: enough resolution to
#: see a shift, few enough that each bin holds a usable number of rows.
NUMERIC_BINS = 10

#: Categories rarer than this in the reference are pooled into one bucket, so a
#: long tail of rare values cannot manufacture drift out of sampling noise.
RARE_CATEGORY_FLOOR = 0.01


@dataclass
class FeatureDrift:
    """One feature's verdict."""

    feature: str
    kind: str
    psi: float
    statistic: float
    statistic_name: str
    p_value: float | None
    drifted: bool
    severity: str
    reference_summary: dict[str, float]
    current_summary: dict[str, float]

    def describe(self) -> str:
        mark = "DRIFT" if self.drifted else "ok   "
        return (
            f"[{mark}] {self.feature:<32} psi={self.psi:.4f} "
            f"{self.statistic_name}={self.statistic:.4f} ({self.severity})"
        )


@dataclass
class DriftReport:
    """Every feature's verdict, and the one decision drawn from them."""

    reference_rows: int
    current_rows: int
    feature_set: str
    threshold_psi: float
    min_drifted_features: int
    features: list[FeatureDrift] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def drifted_features(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.drifted]

    @property
    def n_drifted(self) -> int:
        return len(self.drifted_features)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / len(self.features) if self.features else 0.0

    @property
    def max_psi(self) -> float:
        return max((f.psi for f in self.features), default=0.0)

    @property
    def drift_detected(self) -> bool:
        """Whether enough features moved to call this a drifted population.

        A single feature over the threshold is not enough. One feature can move
        for a mundane reason -- a changed code list, a seasonal effect -- and
        retraining on every such blip would churn the registry for nothing. The
        configured count is what makes this a population-level statement.
        """
        return self.n_drifted >= self.min_drifted_features

    @property
    def decision(self) -> str:
        """``retrain`` or ``no_action``. Never ``rollback``.

        Drift has no authority to reject a model: it says the inputs changed, not
        that predictions got worse. Rollback belongs to operational signals and
        promotion belongs to the M6 gates.
        """
        return "retrain" if self.drift_detected else "no_action"

    def summary(self) -> str:
        verdict = "DRIFT DETECTED" if self.drift_detected else "no drift"
        return (
            f"[{verdict}] {self.n_drifted}/{len(self.features)} features over "
            f"psi={self.threshold_psi} (max {self.max_psi:.4f}); "
            f"decision={self.decision}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "drift_detected": self.drift_detected,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "feature_set": self.feature_set,
            "thresholds": {
                "psi": self.threshold_psi,
                "min_drifted_features": self.min_drifted_features,
            },
            "n_features": len(self.features),
            "n_drifted": self.n_drifted,
            "share_drifted": round(self.share_drifted, 6),
            "max_psi": round(self.max_psi, 6),
            "drifted_features": [f.feature for f in self.drifted_features],
            "features": [asdict(f) for f in self.features],
            "metadata": self.metadata,
        }


def _severity(psi: float) -> str:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MINOR:
        return "minor"
    return "none"


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI between two proportion vectors of the same length."""
    ref = np.clip(reference, EPSILON, None)
    cur = np.clip(current, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _numeric_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    """Quantile-binned PSI, corroborated by a two-sample KS test.

    Bin edges come from the *reference* and are then applied to both samples.
    Deriving them from the combined data would let the current window move the
    ruler it is being measured against.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)

    if ref.size == 0 or cur.size == 0:
        return FeatureDrift(
            feature=name,
            kind="numeric",
            psi=0.0,
            statistic=0.0,
            statistic_name="ks",
            p_value=None,
            drifted=False,
            severity="none",
            reference_summary={"n": float(ref.size)},
            current_summary={"n": float(cur.size)},
        )

    quantiles = np.linspace(0, 1, NUMERIC_BINS + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # A constant feature in the reference: PSI is undefined, so fall back to
        # asking whether the current window is still constant at the same value.
        moved = not np.allclose(cur, ref[0])
        return FeatureDrift(
            feature=name,
            kind="numeric",
            psi=float(PSI_MAJOR if moved else 0.0),
            statistic=float(moved),
            statistic_name="constant_shift",
            p_value=None,
            drifted=moved,
            severity=_severity(PSI_MAJOR if moved else 0.0),
            reference_summary={"n": float(ref.size), "constant": float(ref[0])},
            current_summary={"n": float(cur.size)},
        )

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0] / ref.size
    cur_counts = np.histogram(cur, bins=edges)[0] / cur.size
    psi = _psi(ref_counts, cur_counts)

    from scipy.stats import ks_2samp

    ks = ks_2samp(ref, cur)

    return FeatureDrift(
        feature=name,
        kind="numeric",
        psi=psi,
        statistic=float(ks.statistic),
        statistic_name="ks",
        p_value=float(ks.pvalue),
        drifted=False,
        severity=_severity(psi),
        reference_summary={
            "n": float(ref.size),
            "mean": float(np.mean(ref)),
            "std": float(np.std(ref)),
            "p50": float(np.median(ref)),
        },
        current_summary={
            "n": float(cur.size),
            "mean": float(np.mean(cur)),
            "std": float(np.std(cur)),
            "p50": float(np.median(cur)),
        },
    )


def _categorical_drift(name: str, reference: pd.Series, current: pd.Series) -> FeatureDrift:
    """PSI over category proportions, corroborated by a chi-square test.

    Categories are taken from the reference, with everything else pooled into
    ``__other__``. A value the model never saw is drift, but a hundred distinct
    unseen values are one fact, not a hundred.
    """
    ref = reference.astype("string").fillna("UNK")
    cur = current.astype("string").fillna("UNK")

    proportions = ref.value_counts(normalize=True)
    keep = list(proportions[proportions >= RARE_CATEGORY_FLOOR].index)

    def _folded(series: pd.Series) -> pd.Series:
        return series.where(series.isin(keep), other="__other__")

    categories = [*keep, "__other__"]
    ref_counts = _folded(ref).value_counts().reindex(categories, fill_value=0)
    cur_counts = _folded(cur).value_counts().reindex(categories, fill_value=0)

    ref_share = (ref_counts / max(len(ref), 1)).to_numpy(dtype=float)
    cur_share = (cur_counts / max(len(cur), 1)).to_numpy(dtype=float)
    psi = _psi(ref_share, cur_share)

    # Chi-square against the reference shape, scaled to the current sample size.
    expected = np.clip(ref_share * len(cur), EPSILON, None)
    observed = cur_counts.to_numpy(dtype=float)
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    degrees = max(len(categories) - 1, 1)
    from scipy.stats import chi2

    p_value = float(chi2.sf(statistic, degrees))

    return FeatureDrift(
        feature=name,
        kind="categorical",
        psi=psi,
        statistic=statistic,
        statistic_name="chi2",
        p_value=p_value,
        drifted=False,
        severity=_severity(psi),
        reference_summary={"n": float(len(ref)), "n_categories": float(len(keep))},
        current_summary={"n": float(len(cur)), "n_categories": float(cur.nunique())},
    )


def compare_frames(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_set: str = "engineered",
    threshold_psi: float = PSI_MINOR,
    min_drifted_features: int = 2,
    metadata: dict[str, Any] | None = None,
) -> DriftReport:
    """Compare two register-shaped frames feature by feature.

    Both sides go through the same :func:`build_features` code the model trains
    on, so drift is measured on what the model actually sees rather than on the
    raw columns.
    """
    reference_features = add_engineered_features(reference)
    current_features = add_engineered_features(current)
    numeric, categorical = feature_columns(feature_set)

    results: list[FeatureDrift] = []
    for column in numeric:
        result = _numeric_drift(column, reference_features[column], current_features[column])
        result.drifted = result.psi >= threshold_psi
        results.append(result)
    for column in categorical:
        result = _categorical_drift(column, reference_features[column], current_features[column])
        result.drifted = result.psi >= threshold_psi
        results.append(result)

    report = DriftReport(
        reference_rows=len(reference),
        current_rows=len(current),
        feature_set=feature_set,
        threshold_psi=threshold_psi,
        min_drifted_features=min_drifted_features,
        features=results,
        metadata=metadata or {},
    )
    LOGGER.info(report.summary())
    return report


def compare_with_config(
    config: Config,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> DriftReport:
    """:func:`compare_frames` with the thresholds the project configured."""
    return compare_frames(
        reference,
        current,
        feature_set=config.drift_feature_set,
        threshold_psi=config.drift_threshold_psi,
        min_drifted_features=config.drift_min_features,
        metadata=metadata,
    )
