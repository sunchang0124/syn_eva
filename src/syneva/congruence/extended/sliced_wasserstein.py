from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy import stats

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class SlicedWasserstein:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="sliced_wasserstein",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nums = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]
        if not nums:
            return MetricResult(
                spec=self.spec,
                skip_reason="There are no numeric columns to compare.",
            )
        X_real = real[nums].to_numpy(dtype=float)
        X_syn = synthetic[nums].to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        d = X_real.shape[1]
        n_proj = 50
        dirs = rng.normal(size=(n_proj, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        dists = [stats.wasserstein_distance(X_real @ v, X_syn @ v) for v in dirs]
        sw = float(np.mean(dists))
        score = float(max(0.0, 1.0 - sw / 3.0))
        return MetricResult(spec=self.spec, scalars={"score": score, "sliced_wasserstein": sw})
