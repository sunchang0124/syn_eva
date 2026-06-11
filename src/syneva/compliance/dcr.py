from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class DCR:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="dcr",
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
                scalars={"score": 1.0, "median_dcr": 0.0, "p05_dcr": 0.0},
                notes=["no encodable columns"],
            )
        nn = NearestNeighbors(n_neighbors=1).fit(X_real)
        dists, _ = nn.kneighbors(X_syn, return_distance=True)
        dists = dists.ravel()
        median = float(np.median(dists))
        p05 = float(np.quantile(dists, 0.05))
        # Higher distance to the nearest real record = safer. The 5th-percentile
        # distance (worst-case proximity) drives the score, soft-capped at 1.0.
        score = float(min(1.0, p05))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "median_dcr": median, "p05_dcr": p05},
        )
