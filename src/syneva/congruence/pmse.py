from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class PMSE:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="pmse",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        if len(real) == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="A dataset is empty, so there are no rows to classify.",
            )
        X_real, X_syn = encode_pair(real, synthetic, meta)
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        # encode_pair scales only the numeric columns; scale the one-hot columns too
        X = StandardScaler().fit_transform(X)
        # A classifier failure propagates; the runner records it as a metric error.
        clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        c = len(X_syn) / len(X)
        pmse = float(np.mean((p - c) ** 2))
        # A perfect classifier (p = 0 on real rows, 1 on synthetic) attains the
        # maximum pMSE of c(1 - c) (Snoke et al., 2018), so normalizing by it
        # maps fully separable data to 0 whatever the real/synthetic ratio.
        score = float(np.clip(1.0 - pmse / (c * (1.0 - c)), 0.0, 1.0))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "pmse": pmse},
        )
