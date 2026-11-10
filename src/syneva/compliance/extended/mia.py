from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
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

    def __init__(
        self,
        holdout: pd.DataFrame | None = None,
        neighbors: Neighbors | None = None,
        random_state: int = 42,
    ) -> None:
        self.holdout = holdout
        # Shared by the runner; checked and otherwise built in compute (Neighbors.resolve).
        # Always euclidean: the attack has no distance option.
        self.neighbors = neighbors
        self.random_state = random_state

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
        nb = Neighbors.resolve(
            self.neighbors,
            real,
            synthetic,
            meta,
            holdout=self.holdout,
            random_state=self.random_state,
        )
        if nb.n_features == 0 or len(synthetic) < 1 or len(self.holdout) < 1 or len(real) < 1:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "There are no numeric or categorical columns to attack on, or the real, "
                    "synthetic or holdout data is empty."
                ),
            )
        d_mem = nb.real_to_syn(1)
        d_non = nb.holdout_to_syn(1)
        y = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
        scores = -np.concatenate([d_mem, d_non])  # closer => more likely member
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.5
        score = float(1.0 - max(0.0, auc - 0.5) * 2)  # AUC=0.5 => 1; AUC=1 => 0
        return MetricResult(spec=self.spec, scalars={"score": score, "mia_auc": auc})
