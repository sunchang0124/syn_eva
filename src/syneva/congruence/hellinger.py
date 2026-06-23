from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_N_BINS = 20


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


@registry.register
class Hellinger:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="hellinger",
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
            if cm.dtype is ColumnType.CATEGORICAL:
                r = real[name].astype("string").fillna("__NA__").value_counts(normalize=True)
                s = synthetic[name].astype("string").fillna("__NA__").value_counts(normalize=True)
                cats = r.index.union(s.index)
                p = np.array([r.get(c, 0.0) for c in cats])
                q = np.array([s.get(c, 0.0) for c in cats])
                per_column[name] = {"hellinger": _hellinger(p, q)}
            elif cm.dtype is ColumnType.NUMERIC:
                r = pd.to_numeric(real[name], errors="coerce").dropna().to_numpy()
                s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
                if len(r) == 0 or len(s) == 0:
                    notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                    continue
                lo = float(min(r.min(), s.min()))
                hi = float(max(r.max(), s.max()))
                if hi <= lo:
                    per_column[name] = {"hellinger": 0.0}
                    continue
                bins = np.linspace(lo, hi, _N_BINS + 1)
                pr, _ = np.histogram(r, bins=bins)
                ps, _ = np.histogram(s, bins=bins)
                p = pr / pr.sum()
                q = ps / ps.sum()
                per_column[name] = {"hellinger": _hellinger(p, q)}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric or categorical columns"],
            )
        mean_h = sum(v["hellinger"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_h)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_hellinger": float(mean_h)},
            per_column=per_column,
            notes=notes,
        )
