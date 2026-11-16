from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_QUANTILES = np.arange(0.05, 1.0, 0.05)


def _skip_reason(constant: list[str]) -> str:
    if constant:
        names = ", ".join(f"'{n}'" for n in constant)
        return f"No numeric column can be compared; skipped as constant in the real data: {names}."
    return "No numeric column has values to compare."


@registry.register
class QuantileMSE:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="quantile_mse",
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
        constant: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna().to_numpy()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            if r.min() == r.max():
                # No spread in the real column to standardise by.
                notes.append(f"column '{name}' skipped (constant in the real data)")
                constant.append(name)
                continue
            scale = float(np.std(r))
            qr = np.quantile(r, _QUANTILES)
            qs = np.quantile(s, _QUANTILES)
            qmse = float(np.mean(((qr - qs) / scale) ** 2))
            per_column[name] = {"quantile_mse": qmse}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason=_skip_reason(constant),
            )
        mean_qmse = sum(v["quantile_mse"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - np.sqrt(mean_qmse))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_quantile_mse": float(mean_qmse)},
            per_column=per_column,
            notes=notes,
        )
