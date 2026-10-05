from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

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

    def __init__(self, distance: str = "euclidean", holdout: pd.DataFrame | None = None) -> None:
        self.distance = distance
        self.holdout = holdout

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance, holdout=self.holdout)
        if nb.n_features == 0 or len(real) == 0 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["insufficient data"],
            )
        d_real = nb.syn_to_real(1)
        d_syn = nb.syn_self(1)
        ratio_syn = d_real / np.where(d_syn > 0, d_syn, 1e-9)
        med = float(np.median(ratio_syn))
        notes = ["capped for gower"] if nb.capped else []
        if self.holdout is not None and len(self.holdout) > 1:
            hs = nb.holdout_self(1)
            ratio_hold = nb.holdout_to_real(1) / np.where(hs > 0, hs, 1e-9)
            med_hold = float(np.median(ratio_hold))
            score = float(min(1.0, max(0.0, med / (med_hold + 1e-12))))
            return MetricResult(
                spec=self.spec,
                scalars={"score": score, "nndr_median": med, "nndr_median_holdout": med_hold},
                notes=notes,
            )
        score = float(min(1.0, med))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nndr_median": med},
            notes=notes,
        )
