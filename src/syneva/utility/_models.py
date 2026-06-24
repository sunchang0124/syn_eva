from __future__ import annotations

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from syneva.core.metadata import ColumnType, Metadata


def _build_preprocessor(meta: Metadata, features: list[str]) -> ColumnTransformer:
    num = [f for f in features if meta.columns[f].dtype is ColumnType.NUMERIC]
    cat = [
        f for f in features if meta.columns[f].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )


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


def select_features(meta: Metadata, target: str, override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    return [n for n, cm in meta.columns.items() if n != target and cm.dtype is not ColumnType.ID]
