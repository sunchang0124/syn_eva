from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry


def _ratio(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    # d2 == 0 implies d1 == 0: the row copies a duplicated real record, so its
    # ratio is 0 (maximally close) rather than 0/0.
    return np.divide(d1, d2, out=np.zeros_like(d1, dtype=float), where=d2 > 0)


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
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["insufficient data"],
            )
        # Conventional NNDR: distance to the nearest real record over distance to
        # the second-nearest real record. Synthetic-to-synthetic distances must
        # not enter, or duplicated synthetic rows inflate the ratio.
        ratio_syn = _ratio(nb.syn_to_real(1), nb.syn_to_real(2))
        med = float(np.median(ratio_syn))
        notes = ["capped for gower"] if nb.capped else []
        if self.holdout is not None and len(self.holdout) > 0:
            ratio_hold = _ratio(nb.holdout_to_real(1), nb.holdout_to_real(2))
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
