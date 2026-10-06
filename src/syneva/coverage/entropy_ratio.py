from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * p.apply(lambda v: math.log(v) if v > 0 else 0.0)).sum())


@registry.register
class EntropyRatio:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="entropy_ratio",
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

            real_series = real[name].astype("string").fillna("__NA__")
            syn_series = synthetic[name].astype("string").fillna("__NA__")

            h_real = _entropy(real_series)
            h_syn = _entropy(syn_series)

            if h_real == 0.0:
                # Constant column — ratio undefined; skip with a note
                notes.append(
                    f"column '{name}' skipped (H(real)=0, constant column — ratio undefined)"
                )
                continue

            raw_ratio = h_syn / h_real
            # Clamp to [0, 1]: ratio > 1 means over-dispersion, capped at 1
            ratio = float(min(1.0, max(0.0, raw_ratio)))

            per_column[name] = {
                "ratio": ratio,
                "entropy_real": h_real,
                "entropy_synthetic": h_syn,
            }

        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No categorical column has a defined entropy ratio.",
            )

        score = float(sum(v["ratio"] for v in per_column.values()) / len(per_column))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score},
            per_column=per_column,
            notes=notes,
        )
