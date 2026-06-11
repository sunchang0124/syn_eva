from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class NNDR:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="nndr",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] == 0 or X_real.shape[0] == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["no encodable columns"],
            )
        nn_real = NearestNeighbors(n_neighbors=1).fit(X_real)
        d_real, _ = nn_real.kneighbors(X_syn)
        nn_syn = NearestNeighbors(n_neighbors=2).fit(X_syn)  # 2: nearest is self
        d_syn, _ = nn_syn.kneighbors(X_syn)
        d_syn = d_syn[:, 1]
        ratio = d_real.ravel() / np.where(d_syn > 0, d_syn, 1e-9)
        med = float(np.median(ratio))
        score = float(min(1.0, med))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nndr_median": med},
        )
