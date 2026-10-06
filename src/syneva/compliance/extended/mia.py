from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_frames
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
        if self.holdout is None:
            # Every real row trained the generator, so splitting `real` yields no true
            # non-members and the attack AUC is ~0.5 regardless of leakage. Skip rather
            # than report a perfect score; scalars=None keeps it out of aggregates.
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "No holdout data was provided. The attack compares how close the "
                    "synthetic data is to training rows versus real rows the generator "
                    "never saw, so it needs a holdout of such unseen rows. Without one "
                    "it would always report no leakage, even for a verbatim copy. "
                    "Provide a holdout (holdout=... in Python, --holdout in the CLI, or "
                    "the holdout upload in the UI) to run it."
                ),
            )
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
        scores = -np.concatenate([d_mem.ravel(), d_non.ravel()])  # closer => more likely member
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.5
        score = float(1.0 - max(0.0, auc - 0.5) * 2)  # AUC=0.5 => 1; AUC=1 => 0
        return MetricResult(spec=self.spec, scalars={"score": score, "mia_auc": auc})
