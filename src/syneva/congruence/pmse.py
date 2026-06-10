from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _encode(df: pd.DataFrame, meta: Metadata) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name, cmeta in meta.columns.items():
        if cmeta.dtype is ColumnType.NUMERIC:
            col = pd.to_numeric(df[name], errors="coerce").fillna(0).to_numpy()
            pieces.append(col.reshape(-1, 1))
        elif cmeta.dtype is ColumnType.CATEGORICAL:
            dummies = pd.get_dummies(df[name].astype("string").fillna("__NA__"), dtype=float)
            pieces.append(dummies.to_numpy())
    return np.concatenate(pieces, axis=1) if pieces else np.zeros((len(df), 0))


@registry.register
class PMSE:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="pmse",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real = _encode(real, meta)
        X_syn = _encode(synthetic, meta)
        # align column counts (categorical-induced widening)
        cols = max(X_real.shape[1], X_syn.shape[1])
        X_real = np.pad(X_real, ((0, 0), (0, cols - X_real.shape[1])))
        X_syn = np.pad(X_syn, ((0, 0), (0, cols - X_syn.shape[1])))
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        X = StandardScaler().fit_transform(X)
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
            p = clf.predict_proba(X)[:, 1]
        except Exception as exc:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "pmse": 0.0},
                notes=[f"pMSE classifier failed: {exc}"],
            )
        c = len(X_syn) / len(X)
        pmse = float(np.mean((p - c) ** 2))
        score = float(np.clip(1.0 - pmse * 8, 0.0, 1.0))  # rescale; pmse near 0 => score near 1
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "pmse": pmse},
        )
