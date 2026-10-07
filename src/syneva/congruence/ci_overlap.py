from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12


@registry.register
class CIOverlap:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="ci_overlap",
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
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) < 2 or len(s) < 2:
                notes.append(f"column '{name}' skipped (need >=2 values)")
                continue
            se_r = float(r.std()) / math.sqrt(len(r))
            se_s = float(s.std()) / math.sqrt(len(s))
            lo_r, hi_r = float(r.mean()) - 1.96 * se_r, float(r.mean()) + 1.96 * se_r
            lo_s, hi_s = float(s.mean()) - 1.96 * se_s, float(s.mean()) + 1.96 * se_s
            avg_width = ((hi_r - lo_r) + (hi_s - lo_s)) / 2.0
            if avg_width < _EPS:
                # Zero-width CIs (constant column): full overlap iff the means match.
                frac = 1.0 if abs(float(r.mean()) - float(s.mean())) < _EPS else 0.0
            else:
                overlap = max(0.0, min(hi_r, hi_s) - max(lo_r, lo_s))
                frac = float(min(1.0, max(0.0, overlap / avg_width)))
            per_column[name] = {"ci_overlap": frac}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No numeric column has enough values to compute a confidence interval.",
            )
        mean_overlap = sum(v["ci_overlap"] for v in per_column.values()) / len(per_column)
        return MetricResult(
            spec=self.spec,
            scalars={"score": float(mean_overlap), "mean_ci_overlap": float(mean_overlap)},
            per_column=per_column,
            notes=notes,
        )
