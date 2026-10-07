from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12
_CAP = 500


def _subsample(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(x) > _CAP:
        idx = rng.choice(len(x), _CAP, replace=False)
        return x[idx]
    return x


@registry.register
class MMD:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="mmd",
        c="congruence",
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
        if x_real.shape[1] == 0 or len(x_real) == 0 or len(x_syn) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "There are no numeric or categorical columns to measure distances on, or a "
                    "dataset is empty."
                ),
            )
        rng = np.random.default_rng(42)
        x_real = _subsample(x_real, rng)
        x_syn = _subsample(x_syn, rng)
        z = np.vstack([x_real, x_syn])
        dists = euclidean_distances(z)
        upper = dists[np.triu_indices(len(z), k=1)]
        med = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
        gamma = 1.0 / (2.0 * med**2 + _EPS)
        k_xx = rbf_kernel(x_real, x_real, gamma=gamma)
        k_yy = rbf_kernel(x_syn, x_syn, gamma=gamma)
        k_xy = rbf_kernel(x_real, x_syn, gamma=gamma)
        mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        mmd = float(np.sqrt(max(0.0, mmd2)))
        score = float(min(1.0, max(0.0, 1.0 - mmd)))
        return MetricResult(spec=self.spec, scalars={"score": score, "mmd": mmd})
