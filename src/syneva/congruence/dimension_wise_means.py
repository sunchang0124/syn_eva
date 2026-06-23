from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12


@registry.register
class DimensionWiseMeans:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="dimension_wise_means",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            d = abs(float(r.mean()) - float(s.mean())) / (float(r.std()) + _EPS)
            per_column[name] = {"std_mean_diff": d}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )
        mean_d = sum(v["std_mean_diff"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_d)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_abs_std_diff": float(mean_d)},
            per_column=per_column,
            notes=notes,
        )
