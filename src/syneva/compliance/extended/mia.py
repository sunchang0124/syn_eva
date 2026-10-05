from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_frames, encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class MembershipInferenceAttack:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="mia_auc",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, holdout: pd.DataFrame | None = None) -> None:
        self.holdout = holdout

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        if self.holdout is not None:
            x_real, x_syn, x_hold = encode_frames([real, synthetic, self.holdout], meta)
            if x_real.shape[1] == 0 or len(x_syn) < 1 or len(x_hold) < 1 or len(x_real) < 1:
                return MetricResult(
                    spec=self.spec,
                    scalars={"score": 1.0, "mia_auc": 0.5},
                    notes=["insufficient data"],
                )
            nn = NearestNeighbors(n_neighbors=1).fit(x_syn)
            d_mem, _ = nn.kneighbors(x_real)
            d_non, _ = nn.kneighbors(x_hold)
            y = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_hold))])
            scores = -np.concatenate([d_mem.ravel(), d_non.ravel()])
        else:
            X_real, X_syn = encode_pair(real, synthetic, meta)
            if X_real.shape[1] == 0 or len(X_real) < 4:
                return MetricResult(
                    spec=self.spec,
                    scalars={"score": 1.0, "mia_auc": 0.5},
                    notes=["insufficient data"],
                )
            members, non_members = train_test_split(X_real, test_size=0.5, random_state=42)
            nn = NearestNeighbors(n_neighbors=1).fit(X_syn)
            d_mem, _ = nn.kneighbors(members)
            d_non, _ = nn.kneighbors(non_members)
            y = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
            scores = -np.concatenate([d_mem.ravel(), d_non.ravel()])  # closer => more likely member
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.5
        score = float(1.0 - max(0.0, auc - 0.5) * 2)  # AUC=0.5 => 1; AUC=1 => 0
        return MetricResult(spec=self.spec, scalars={"score": score, "mia_auc": auc})
