from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
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

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        # Encode real + synthetic together so the two clouds share one space.
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "alpha_precision": 1.0, "beta_recall": 1.0},
                notes=["no encodable columns"],
            )
        k = max(1, min(5, len(X_real) - 1))
        nn_real = NearestNeighbors(n_neighbors=k + 1).fit(X_real)
        d_real, _ = nn_real.kneighbors(X_real)
        radii_real = d_real[:, -1]
        nn_syn = NearestNeighbors(n_neighbors=k + 1).fit(X_syn)
        d_syn, _ = nn_syn.kneighbors(X_syn)
        radii_syn = d_syn[:, -1]

        # alpha-precision: fraction of synth inside the alpha-ball of some real point.
        d_to_real = nn_real.kneighbors(X_syn, n_neighbors=1, return_distance=True)[0].ravel()
        alpha = np.quantile(radii_real, 0.9)
        precision = float(np.mean(d_to_real <= alpha))

        # beta-recall: fraction of real inside the beta-ball of some synth point.
        d_to_syn = nn_syn.kneighbors(X_real, n_neighbors=1, return_distance=True)[0].ravel()
        beta = np.quantile(radii_syn, 0.9)
        recall = float(np.mean(d_to_syn <= beta))

        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float((precision + recall) / 2),
                "alpha_precision": precision,
                "beta_recall": recall,
            },
        )
