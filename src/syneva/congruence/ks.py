from __future__ import annotations

from typing import ClassVar

import pandas as pd
from scipy import stats

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class KSStatistic:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="ks_statistic",
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

        for name, cmeta in meta.columns.items():
            if cmeta.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(s) == 0 or len(r) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            stat, p = stats.ks_2samp(r.values, s.values)
            per_column[name] = {"ks_statistic": float(stat), "p_value": float(p)}

        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )

        mean_ks = sum(v["ks_statistic"] for v in per_column.values()) / len(per_column)
        score = float(1.0 - min(1.0, max(0.0, mean_ks)))

        from syneva.render.figures import HistogramOverlay

        worst = max(per_column.items(), key=lambda kv: kv[1]["ks_statistic"])[0]
        payload = HistogramOverlay(
            real=pd.to_numeric(real[worst], errors="coerce").dropna().tolist(),
            synthetic=pd.to_numeric(synthetic[worst], errors="coerce").dropna().tolist(),
            title=f"KS overlay: {worst}",
        )
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_ks_statistic": mean_ks},
            per_column=per_column,
            plot_payload=payload,
            notes=notes,
        )
