from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


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
                scalars={"score": 1.0, "novelty_rate": 1.0, "duplicates": 0.0},
                notes=["synthetic dataset is empty; novelty_rate defaulted to 1.0"],
            )

        real_set: set[tuple] = set(map(tuple, real.values.tolist()))
        novel = sum(1 for row in synthetic.values.tolist() if tuple(row) not in real_set)
        n_syn = len(synthetic)
        duplicates = float(n_syn - novel)
        rate = float(novel / n_syn)
        # clamp to [0, 1] to guard against floating-point edge cases
        rate = min(1.0, max(0.0, rate))

        return MetricResult(
            spec=self.spec,
            scalars={"score": rate, "novelty_rate": rate, "duplicates": duplicates},
        )
