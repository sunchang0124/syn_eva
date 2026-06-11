from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from syneva.core.metadata import ColumnType, Metadata


def build_pipeline(task_type: str, meta: Metadata, features: list[str], random_state: int):
    num = [f for f in features if meta.columns[f].dtype is ColumnType.NUMERIC]
    cat = [
        f for f in features if meta.columns[f].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]
    pre = ColumnTransformer(
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
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    return Pipeline([("pre", pre), ("model", model)])


def select_features(meta: Metadata, target: str, override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    return [n for n, cm in meta.columns.items() if n != target and cm.dtype is not ColumnType.ID]
