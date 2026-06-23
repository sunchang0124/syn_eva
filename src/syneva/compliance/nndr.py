from __future__ import annotations

from typing import ClassVar

import numpy as np

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
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

    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["insufficient data"],
            )
        d_real = nb.syn_to_real(1)
        d_syn = nb.syn_self(1)
        ratio = d_real / np.where(d_syn > 0, d_syn, 1e-9)
        med = float(np.median(ratio))
        score = float(min(1.0, med))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nndr_median": med},
            notes=["capped for gower"] if nb.capped else [],
        )
