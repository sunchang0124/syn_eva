from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from syneva.core.errors import MetricError
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
        if not 0.0 < self.tail_quantile < 0.5:
            raise MetricError("tail_quantile must be in (0, 0.5)")
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        lowers: list[float] = []
        uppers: list[float] = []
        num_cols = numeric_columns(meta)
        if not num_cols:
            return MetricResult(
                spec=self.spec,
                skip_reason="There are no numeric columns with tails to cover.",
            )
        q = self.tail_quantile
        processed_cols = 0
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
            processed_cols += 1
            col_tails: list[float] = []
            if lo == float(rv.min()):
                notes.append(f"column '{name}' lower tail skipped (cutoff equals minimum)")
            else:
                lower = min(float((sv < lo).mean()) / q, 1.0)
                lowers.append(lower)
                col_tails.append(lower)
            if hi == float(rv.max()):
                notes.append(f"column '{name}' upper tail skipped (cutoff equals maximum)")
            else:
                upper = min(float((sv > hi).mean()) / q, 1.0)
                uppers.append(upper)
                col_tails.append(upper)
            if col_tails:
                per_column[name] = {"tail_coverage": sum(col_tails) / len(col_tails)}
        all_tails = lowers + uppers
        if not all_tails:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "No numeric column has values to measure tails on."
                    if processed_cols == 0
                    else "No numeric column has a tail beyond its minimum or maximum to measure."
                ),
            )
        score = float(min(1.0, max(0.0, sum(all_tails) / len(all_tails))))
        scalars: dict[str, float] = {"score": score}
        if lowers:
            scalars["lower_tail_coverage"] = sum(lowers) / len(lowers)
        if uppers:
            scalars["upper_tail_coverage"] = sum(uppers) / len(uppers)
        return MetricResult(
            spec=self.spec,
            scalars=scalars,
            per_column=per_column,
            notes=notes,
        )
