from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry


@registry.register
class EpsilonIdentifiability:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="epsilon_identifiability",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) < 1:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "identifiability_risk": 0.0},
                notes=["insufficient data"],
            )
        d_self = nb.real_self(1)
        d_syn = nb.real_to_syn(1)
        risk = float(np.mean(d_syn <= d_self))
        score = float(min(1.0, max(0.0, 1.0 - risk)))
        notes = ["unweighted distance; entropy-weighting is a planned refinement"]
        if nb.capped:
            notes.append("capped for gower")
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "identifiability_risk": risk},
            notes=notes,
        )
