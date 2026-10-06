from __future__ import annotations

from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class KAnonymity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="k_anonymity",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=False,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        qi = [n for n, cm in meta.columns.items() if cm.sensitive]
        if not qi:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "No column is marked sensitive=True, so there are no quasi-identifiers to group"
                    " on."
                ),
            )
        groups = synthetic.groupby(qi, dropna=False).size()
        min_k = int(groups.min()) if len(groups) else 0
        score = float(min(1.0, min_k / 5.0))  # k=5 is the conventional safety floor
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "min_k": float(min_k),
                "unique_groups": float(len(groups)),
            },
        )
