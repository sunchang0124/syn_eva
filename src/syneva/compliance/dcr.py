from __future__ import annotations

from typing import ClassVar

import numpy as np

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
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

    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "median_dcr": 0.0, "p05_dcr": 0.0},
                notes=["no encodable columns"],
            )
        dists = nb.syn_to_real(1)
        median = float(np.median(dists))
        p05 = float(np.quantile(dists, 0.05))
        score = float(min(1.0, p05))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "median_dcr": median, "p05_dcr": p05},
            notes=["capped for gower"] if nb.capped else [],
        )
