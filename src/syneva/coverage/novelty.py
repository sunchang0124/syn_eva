from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.core.rows import comparable_row_keys


@registry.register
class NoveltyRate:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="novelty_rate",
        c="coverage",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None

        if len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="The synthetic data is empty.",
            )

        # Column alignment, NaN normalization and ID-column exclusion live in
        # comparable_row_keys, shared with IdenticalMatchRate.
        real_keys, syn_keys = comparable_row_keys(real, synthetic, meta)
        real_set: set[tuple] = set(real_keys)
        novel = sum(1 for t in syn_keys if t not in real_set)
        n_syn = len(synthetic)
        duplicates = float(n_syn - novel)
        rate = float(novel / n_syn)
        # clamp to [0, 1] to guard against floating-point edge cases
        rate = min(1.0, max(0.0, rate))

        return MetricResult(
            spec=self.spec,
            scalars={"score": rate, "novelty_rate": rate, "duplicates": duplicates},
        )
