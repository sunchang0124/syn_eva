from __future__ import annotations

from typing import ClassVar

import numpy as np

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry


@registry.register
class AlphaPrecisionBetaRecall:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="alpha_precision_beta_recall",
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
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "alpha_precision": 1.0, "beta_recall": 1.0},
                notes=["no encodable columns or insufficient rows"],
            )
        k = max(1, min(5, min(len(real), len(synthetic)) - 1))
        radii_real = nb.real_self(k)
        radii_syn = nb.syn_self(k)
        d_to_real = nb.syn_to_real(1)
        alpha = np.quantile(radii_real, 0.9)
        precision = float(np.mean(d_to_real <= alpha))
        d_to_syn = nb.real_to_syn(1)
        beta = np.quantile(radii_syn, 0.9)
        recall = float(np.mean(d_to_syn <= beta))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float((precision + recall) / 2),
                "alpha_precision": precision,
                "beta_recall": recall,
            },
            notes=["capped for gower"] if nb.capped else [],
        )
