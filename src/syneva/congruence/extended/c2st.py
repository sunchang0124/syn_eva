from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_N_SPLITS = 5


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
        # StratifiedKFold needs n_splits members of each class to fill every fold
        if min(len(real), len(synthetic)) < _N_SPLITS:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    f"Each dataset needs at least {_N_SPLITS} rows for {_N_SPLITS}-fold "
                    "cross-validation."
                ),
            )
        # Encode jointly so one-hot columns align by category, not by position.
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="There are no numeric or categorical columns to classify on.",
            )
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        aucs: list[float] = []
        cv = StratifiedKFold(n_splits=_N_SPLITS, shuffle=True, random_state=42)
        for train, test in cv.split(X, y):
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(X[train], y[train])
            aucs.append(float(roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])))
        auc = float(np.mean(aucs))
        # ideal AUC ~ 0.5 (indistinguishable). Score = 1 - 2*|auc - 0.5|.
        score = float(max(0.0, 1.0 - 2 * abs(auc - 0.5)))
        return MetricResult(spec=self.spec, scalars={"score": score, "c2st_auc": auc})
