from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import numeric_columns


@registry.register
@dataclass
class TailCoverage:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="tail_coverage",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tail_quantile: float = 0.05

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        lowers: list[float] = []
        uppers: list[float] = []
        num_cols = numeric_columns(meta)
        if not num_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no numeric columns"]
            )
        q = self.tail_quantile
        for name in num_cols:
            rv = pd.to_numeric(real[name], errors="coerce").dropna()
            sv = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if rv.empty or sv.empty:
                notes.append(f"column '{name}' skipped (no numeric values)")
                continue
            lo, hi = float(rv.quantile(q)), float(rv.quantile(1 - q))
            if lo == hi:
                notes.append(f"column '{name}' skipped (degenerate quantiles)")
                continue
            lower = min(float((sv < lo).mean()) / q, 1.0)
            upper = min(float((sv > hi).mean()) / q, 1.0)
            lowers.append(lower)
            uppers.append(upper)
            per_column[name] = {"tail_coverage": (lower + upper) / 2}
        if not lowers:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no usable numeric columns"],
            )
        mean_lower = sum(lowers) / len(lowers)
        mean_upper = sum(uppers) / len(uppers)
        score = float(min(1.0, max(0.0, (mean_lower + mean_upper) / 2)))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "lower_tail_coverage": mean_lower,
                "upper_tail_coverage": mean_upper,
            },
            per_column=per_column,
            notes=notes,
        )
