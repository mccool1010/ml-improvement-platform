"""Model construction.

The baseline is a regularised logistic regression on the core feature set. It is
intentionally plain: its job is to be a defensible benchmark that a candidate
must beat by a stated margin, not to be the best achievable model.

Both the baseline and the candidate are built as complete scikit-learn pipelines,
so imputation and encoding are fitted on training data only and travel with the
model. That removes the most common source of train/serve skew: preprocessing
that is fitted somewhere the served model cannot see.
"""

from __future__ import annotations

import importlib
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_platform.features.engineering import feature_columns


def _import_estimator(dotted_path: str) -> type:
    """Resolve ``package.module.ClassName`` from configuration."""
    module_name, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name)
    if not isinstance(estimator, type):
        raise TypeError(f"{dotted_path} is not a class")
    return estimator


def build_preprocessor(feature_set: str, *, sparse_ok: bool = True) -> ColumnTransformer:
    """Impute and encode, fitted on training data only.

    Numeric columns get median imputation and scaling. Categorical columns get a
    constant ``UNK`` fill and one-hot encoding, with unseen categories ignored so
    a state or sector absent from training does not crash inference.
    """
    numeric, categorical = feature_columns(feature_set)

    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=25,
                    sparse_output=sparse_ok,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, numeric),
            ("categorical", categorical_steps, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(spec: dict[str, Any]) -> Pipeline:
    """Build a full pipeline from a ``configs/model.yaml`` entry."""
    feature_set = str(spec["feature_set"])
    estimator_class = _import_estimator(str(spec["estimator"]))
    params = dict(spec.get("params") or {})

    # HistGradientBoosting cannot consume a sparse matrix.
    sparse_ok = estimator_class is not HistGradientBoostingClassifier

    return Pipeline(
        [
            ("preprocess", build_preprocessor(feature_set, sparse_ok=sparse_ok)),
            ("estimator", estimator_class(**params)),
        ]
    )


def build_baseline(config_raw: dict[str, Any]) -> Pipeline:
    """Build the baseline model described by the resolved configuration."""
    return build_model(config_raw["baseline"])


def build_candidate(
    config_raw: dict[str, Any], overrides: dict[str, Any] | None = None
) -> Pipeline:
    """Build the candidate model, optionally overriding its hyperparameters."""
    spec = dict(config_raw["candidate"])
    if overrides:
        spec["params"] = {**dict(spec.get("params") or {}), **overrides}
    return build_model(spec)
