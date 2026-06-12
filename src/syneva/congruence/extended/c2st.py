from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from syneva.congruence.pmse import _encode
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class C2ST:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="c2st",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real = _encode(real, meta)
        X_syn = _encode(synthetic, meta)
        cols = max(X_real.shape[1], X_syn.shape[1])
        X_real = np.pad(X_real, ((0, 0), (0, cols - X_real.shape[1])))
        X_syn = np.pad(X_syn, ((0, 0), (0, cols - X_syn.shape[1])))
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        aucs: list[float] = []
        for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(X[train], y[train])
            aucs.append(float(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])))
        auc = float(np.mean(aucs))
        # ideal AUC ~ 0.5 (indistinguishable). Score = 1 - 2*|auc - 0.5|.
        score = float(max(0.0, 1.0 - 2 * abs(auc - 0.5)))
        return MetricResult(spec=self.spec, scalars={"score": score, "c2st_auc": auc})
