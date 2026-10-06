from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class CategoryCoverage:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="category_coverage",
        c="coverage",
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

        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.CATEGORICAL:
                notes.append(f"column '{name}' skipped (not categorical)")
                continue
            real_cats = set(real[name].astype("string").fillna("__NA__"))
            syn_cats = set(synthetic[name].astype("string").fillna("__NA__"))
            if not real_cats:
                notes.append(f"column '{name}' skipped (zero real categories)")
                continue
            cov = len(real_cats & syn_cats) / len(real_cats)
            per_column[name] = {"coverage": float(min(1.0, max(0.0, cov)))}

        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No categorical column has categories to cover.",
            )

        score = float(
            min(1.0, max(0.0, sum(v["coverage"] for v in per_column.values()) / len(per_column)))
        )
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_coverage": score},
            per_column=per_column,
            notes=notes,
        )
