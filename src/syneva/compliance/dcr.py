from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

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

    def __init__(self, distance: str = "euclidean", holdout: pd.DataFrame | None = None) -> None:
        self.distance = distance
        self.holdout = holdout

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance, holdout=self.holdout)
        if nb.n_features == 0 or len(real) == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "There are no numeric or categorical columns to measure distances on, or a "
                    "dataset is empty."
                ),
            )
        dists = nb.syn_to_real(1)
        median = float(np.median(dists))
        p05 = float(np.quantile(dists, 0.05))
        notes = ["capped for gower"] if nb.capped else []
        if self.holdout is not None and len(self.holdout) > 0:
            d_hold = nb.holdout_to_real(1)
            p05_hold = float(np.quantile(d_hold, 0.05))
            median_hold = float(np.median(d_hold))
            score = float(min(1.0, max(0.0, p05 / (p05_hold + 1e-12))))
            return MetricResult(
                spec=self.spec,
                scalars={
                    "score": score,
                    "median_dcr": median,
                    "p05_dcr": p05,
                    "median_dcr_holdout": median_hold,
                    "p05_dcr_holdout": p05_hold,
                },
                notes=notes,
            )
        # Without a holdout, the baseline is how close real records sit to each
        # other (leave-one-out), so the score does not depend on the number of
        # encoded columns or the distance backend.
        if len(real) < 2:
            return MetricResult(
                spec=self.spec,
                skip_reason="Without a holdout, DCR needs at least 2 real rows for a baseline.",
            )
        p05_real = float(np.quantile(nb.real_self(1), 0.05))
        if p05_real == 0.0:
            if p05 == 0.0:
                return MetricResult(
                    spec=self.spec,
                    notes=notes,
                    skip_reason=(
                        "At least 5% of real rows have an exact duplicate, so the real data gives "
                        "no distance baseline; see identical_match_rate for copied rows."
                    ),
                )
            score = 1.0  # synthetic rows sit further from real than real rows do from each other
        else:
            score = float(min(1.0, p05 / p05_real))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "median_dcr": median,
                "p05_dcr": p05,
                "p05_dcr_real": p05_real,
            },
            notes=notes,
        )
