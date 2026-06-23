from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
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

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        x_real, x_syn = encode_pair(real, synthetic, meta)
        if x_real.shape[1] == 0 or len(x_real) < 2 or len(x_syn) < 1:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "identifiability_risk": 0.0},
                notes=["insufficient data"],
            )
        d_self = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
        d_syn = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
        risk = float(np.mean(d_syn < d_self))
        score = float(min(1.0, max(0.0, 1.0 - risk)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "identifiability_risk": risk},
            notes=["unweighted distance; entropy-weighting is a planned refinement"],
        )
