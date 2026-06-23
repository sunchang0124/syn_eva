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
class NNAdversarialAccuracy:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="nn_adversarial_accuracy",
        c="coverage",
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
        if x_real.shape[1] == 0 or len(x_real) < 2 or len(x_syn) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nn_adversarial_accuracy": 0.5},
                notes=["insufficient data"],
            )
        d_rr = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
        d_ss = NearestNeighbors(n_neighbors=2).fit(x_syn).kneighbors(x_syn)[0][:, 1]
        d_rs = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
        d_sr = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(x_syn)[0][:, 0]
        aa = 0.5 * (float(np.mean(d_rs > d_rr)) + float(np.mean(d_sr > d_ss)))
        score = float(min(1.0, max(0.0, 1.0 - 2.0 * abs(aa - 0.5))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nn_adversarial_accuracy": float(aa)},
        )
