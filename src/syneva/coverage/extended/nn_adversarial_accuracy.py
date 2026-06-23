from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry


@registry.register
class NNAdversarialAccuracy:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="nn_adversarial_accuracy",
        c="coverage",
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
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nn_adversarial_accuracy": 0.5},
                notes=["insufficient data"],
            )
        d_rr = nb.real_self(1)
        d_ss = nb.syn_self(1)
        d_rs = nb.real_to_syn(1)
        d_sr = nb.syn_to_real(1)
        aa = 0.5 * (float(np.mean(d_rs > d_rr)) + float(np.mean(d_sr > d_ss)))
        score = float(min(1.0, max(0.0, 1.0 - 2.0 * abs(aa - 0.5))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nn_adversarial_accuracy": float(aa)},
            notes=["capped for gower"] if nb.capped else [],
        )
