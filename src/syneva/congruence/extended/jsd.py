from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.spatial.distance import jensenshannon

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class JensenShannon:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="jsd",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.CATEGORICAL:
                continue
            r = real[name].astype("string").fillna("__NA__").value_counts(normalize=True)
            s = synthetic[name].astype("string").fillna("__NA__").value_counts(normalize=True)
            cats = r.index.union(s.index)
            p = np.array([r.get(c, 0.0) for c in cats])
            q = np.array([s.get(c, 0.0) for c in cats])
            jsd = float(jensenshannon(p, q) ** 2)  # JS distance squared = JS divergence
            per_column[name] = {"jsd": jsd}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=["no categorical columns"],
            )
        mean = float(np.mean([v["jsd"] for v in per_column.values()]))
        return MetricResult(
            spec=self.spec,
            scalars={"score": float(1.0 - mean), "mean_jsd": mean},
            per_column=per_column,
        )
