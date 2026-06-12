from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class DiscriminativeScore:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="discriminative_score",
        c="utility",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        X_real, X_syn = encode_pair(real, synthetic, meta)
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(X_tr, y_tr)
        auc = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        score = float(max(0.0, 1.0 - 2 * abs(auc - 0.5)))
        return MetricResult(spec=self.spec, scalars={"score": score, "auc": auc})
