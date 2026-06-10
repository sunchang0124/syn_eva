from __future__ import annotations

from typing import ClassVar

import pandas as pd
from scipy import stats

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class Wasserstein1:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="wasserstein",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        normalized: list[float] = []

        for name, cmeta in meta.columns.items():
            if cmeta.dtype is not ColumnType.NUMERIC:
                notes.append(f"column '{name}' skipped (not numeric)")
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            col_dist = float(stats.wasserstein_distance(r.values, s.values))
            per_column[name] = {"wasserstein": col_dist}
            scale = max(float(r.std()), 1e-9)
            normalized.append(min(1.0, col_dist / scale))

        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )

        mean_normalized = sum(normalized) / len(normalized)
        score = float(min(1.0, max(0.0, 1.0 - mean_normalized)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_wasserstein": mean_normalized},
            per_column=per_column,
            notes=notes,
        )
