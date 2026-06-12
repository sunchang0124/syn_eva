from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.decomposition import PCA

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.render.figures_pca import PCAScatter


@registry.register
class PCAScatterMetric:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="pca_scatter",
        c="coverage",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=["need >=2 encodable columns"],
            )
        pca = PCA(n_components=2, random_state=42).fit(np.vstack([X_real, X_syn]))
        return MetricResult(
            spec=self.spec,
            scalars={"score": 1.0},
            plot_payload=PCAScatter(
                real_xy=pca.transform(X_real),
                synthetic_xy=pca.transform(X_syn),
                title="PCA scatter (real vs synthetic)",
            ),
        )
