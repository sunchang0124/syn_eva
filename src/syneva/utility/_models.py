from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from syneva.core.metadata import ColumnType, Metadata

# dtypes _build_preprocessor encodes; every other dtype falls to remainder="drop"
_NUMERIC = (ColumnType.NUMERIC,)
_CATEGORICAL = (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)


def _as_float(X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    # nullable dtypes (Int64, Float64) carry pd.NA, which SimpleImputer can't compare
    return X.astype("float64")


def _as_object(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    # SimpleImputer only treats NaN as missing: None and pd.NA would become categories
    # (or crash on comparison), so map every null to NaN first
    X = pd.DataFrame(X)
    return X.astype(object).where(X.notna(), np.nan)


def _build_preprocessor(meta: Metadata, features: list[str]) -> ColumnTransformer:
    num = [f for f in features if meta.columns[f].dtype in _NUMERIC]
    cat = [f for f in features if meta.columns[f].dtype in _CATEGORICAL]
    # keep_empty_features keeps an all-missing column as a constant one, so every
    # input feature always maps to a fixed output block (see source_columns)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("na", FunctionTransformer(_as_float, feature_names_out="one-to-one")),
                        ("imp", SimpleImputer(keep_empty_features=True)),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("na", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
                        (
                            "imp",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )


def source_columns(pre: ColumnTransformer) -> list[str]:
    """Return the original feature behind each output column of a fitted
    `_build_preprocessor`.

    A numeric feature yields one entry and a categorical feature one per one-hot
    category, so the result is as long as the transformed matrix. Columns are matched
    by position, not by the names sklearn reports, and each block is checked against
    the transformer's actual output width.
    """
    out: list[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder" or len(cols) == 0:
            continue
        if name == "cat":
            cats = trans.named_steps["oh"].categories_
            block = [c for c, cs in zip(cols, cats, strict=True) for _ in cs]
        else:
            block = list(cols)
        width = pre.output_indices_[name]
        if len(block) != width.stop - width.start:
            raise RuntimeError(
                f"cannot map '{name}' outputs to features: expected "
                f"{width.stop - width.start} columns, derived {len(block)}"
            )
        out.extend(block)
    return out


def build_pipeline(task_type: str, meta: Metadata, features: list[str], random_state: int):
    pre = _build_preprocessor(meta, features)
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    return Pipeline([("pre", pre), ("model", model)])


def build_panel_pipelines(
    task_type: str, meta: Metadata, features: list[str], random_state: int
) -> dict[str, Pipeline]:
    """Return {model_name: Pipeline} for the utility model panel. Each pipeline
    shares the same preprocessor as build_pipeline; only the final estimator differs."""
    if task_type == "classification":
        estimators = {
            "linear": LogisticRegression(max_iter=1000, random_state=random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            "hist_gbdt": HistGradientBoostingClassifier(random_state=random_state),
        }
    else:
        estimators = {
            "linear": Ridge(random_state=random_state),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            "hist_gbdt": HistGradientBoostingRegressor(random_state=random_state),
        }
    return {
        name: Pipeline([("pre", _build_preprocessor(meta, features)), ("model", est)])
        for name, est in estimators.items()
    }


def select_features(
    meta: Metadata, target: str, override: list[str] | None, notes: list[str] | None = None
) -> list[str]:
    """Return the task's features that the preprocessor can encode, without
    duplicates or the target. Defaults to every other column in ``meta``.

    When an override lists the target or features of another dtype, a note appended
    to ``notes`` says which were left out and why. Duplicates are dropped silently.
    """
    candidates = override if override is not None else list(meta.columns)
    encoded = _NUMERIC + _CATEGORICAL
    unique = list(dict.fromkeys(candidates))
    features = [f for f in unique if f != target and meta.columns[f].dtype in encoded]
    if override is not None and notes is not None:
        if target in unique:
            notes.append(f"task '{target}': the target is listed as a feature; ignored")
        unsupported = [f for f in unique if f != target and f not in features]
        if unsupported:
            names = ", ".join(f"'{f}'" for f in unsupported)
            notes.append(
                f"task '{target}': features {names} ignored (not numeric, categorical or boolean)"
            )
    return features
