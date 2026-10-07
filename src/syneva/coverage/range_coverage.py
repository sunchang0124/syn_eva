from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class RangeCoverage:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="range_coverage",
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
            if cm.dtype is not ColumnType.NUMERIC:
                notes.append(f"column '{name}' skipped (not numeric)")
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            real_min, real_max = float(r.min()), float(r.max())
            syn_min, syn_max = float(s.min()), float(s.max())
            real_range = real_max - real_min
            if real_range == 0.0:
                # Constant real column: synthetic matches if it also equals that constant value
                cov = 1.0 if (syn_min <= real_min <= syn_max) else 0.0
                notes.append(
                    f"column '{name}': constant real column (range=0); "
                    f"coverage={'1.0' if cov == 1.0 else '0.0'}"
                )
                per_column[name] = {"coverage": cov}
                continue
            overlap = max(0.0, min(real_max, syn_max) - max(real_min, syn_min))
            per_column[name] = {"coverage": float(min(1.0, overlap / real_range))}

        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No numeric column has values to cover.",
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
