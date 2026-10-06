from __future__ import annotations

from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.core.rows import comparable_row_keys


@registry.register
class IdenticalMatchRate:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="identical_match_rate",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        real_keys, syn_keys = comparable_row_keys(real, synthetic, meta)
        real_set = set(real_keys)
        matches = sum(1 for t in syn_keys if t in real_set)
        rate = float(matches / len(synthetic)) if len(synthetic) else 0.0
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float(1.0 - rate),
                "identical_match_rate": rate,
                "identical_matches": float(matches),
            },
        )
