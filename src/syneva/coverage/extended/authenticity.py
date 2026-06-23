from __future__ import annotations

from typing import ClassVar

import numpy as np

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry


@registry.register
class Authenticity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="authenticity",
        c="coverage",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "authenticity": 1.0},
                notes=["insufficient data"],
            )
        d_syn_to_real = nb.syn_to_real(1)
        d_real_self = nb.real_self(1)
        eps = 0.05 * float(np.median(d_real_self))
        auth = float(np.mean(d_syn_to_real > eps))
        return MetricResult(
            spec=self.spec,
            scalars={"score": auth, "authenticity": auth},
            notes=["capped for gower"] if nb.capped else [],
        )
