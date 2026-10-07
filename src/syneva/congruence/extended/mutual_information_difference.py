from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _discretize(df: pd.DataFrame, name: str, dtype: ColumnType) -> np.ndarray:
    if dtype is ColumnType.NUMERIC:
        v = pd.to_numeric(df[name], errors="coerce")
        binned = pd.qcut(v, 10, labels=False, duplicates="drop")
        return binned.fillna(-1).astype(int).to_numpy()
    return pd.factorize(df[name].astype("string").fillna("__NA__"))[0]


def _nmi_matrix(df: pd.DataFrame, meta: Metadata, cols: list[str]) -> np.ndarray:
    disc = [_discretize(df, c, meta.columns[c].dtype) for c in cols]
    k = len(cols)
    m = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = float(normalized_mutual_info_score(disc[i], disc[j]))
            m[i, j] = m[j, i] = v
    return m


@registry.register
class MutualInformationDifference:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="mutual_information_difference",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        cols = [
            n
            for n, cm in meta.columns.items()
            if cm.dtype in (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        ]
        if len(cols) < 2:
            return MetricResult(
                spec=self.spec,
                skip_reason="Mutual information needs at least 2 usable columns.",
            )
        mr = _nmi_matrix(real, meta, cols)
        ms = _nmi_matrix(synthetic, meta, cols)
        iu = np.triu_indices(len(cols), k=1)
        diff = float(np.mean(np.abs(mr[iu] - ms[iu])))
        score = float(min(1.0, max(0.0, 1.0 - diff)))
        return MetricResult(spec=self.spec, scalars={"score": score, "mean_mi_diff": diff})
