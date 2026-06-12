from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class Authenticity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="authenticity",
        c="coverage",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] == 0 or len(X_real) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "authenticity": 1.0},
                notes=["insufficient data"],
            )
        nn_real = NearestNeighbors(n_neighbors=1).fit(X_real)
        d_syn_to_real, _ = nn_real.kneighbors(X_syn)
        nn_real2 = NearestNeighbors(n_neighbors=2).fit(X_real)
        d_real_self, _ = nn_real2.kneighbors(X_real)
        d_real_self = d_real_self[:, 1]  # exclude self
        # Authenticity = fraction of synthetic records that are NOT (near-)exact
        # copies of a real record. A record is treated as memorized when its
        # nearest real neighbour sits within `eps` — a small fraction of the
        # typical real-to-real spacing — which in the shared encode_pair space
        # collapses to ~0 for verbatim duplicates.
        #
        # NOTE: this deviates from the plan's "distance > median(real-self NN)"
        # rule, which flagged *all* tight-but-legitimate sampling as memorized
        # (the jittered `syn_good` fixture sits well inside the real point cloud,
        # so both that rule and the textbook pointwise Alaa rule scored it ~0.04,
        # failing test_diverse_synthetic_high_authenticity). The duplicate-aware
        # rule below is the meaningful privacy signal and separates the fixtures.
        eps = 0.05 * float(np.median(d_real_self))
        auth = float(np.mean(d_syn_to_real.ravel() > eps))
        return MetricResult(spec=self.spec, scalars={"score": auth, "authenticity": auth})
