from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class TotalVariationDistance:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="tvd",
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
            if cmeta.dtype is not ColumnType.CATEGORICAL:
                continue
            r = real[name].astype("string").fillna("__NA__")
            s = synthetic[name].astype("string").fillna("__NA__")
            p_real = r.value_counts(normalize=True)
            p_syn = s.value_counts(normalize=True)
            cats = p_real.index.union(p_syn.index)
            col_tvd = float(0.5 * sum(abs(p_real.get(c, 0.0) - p_syn.get(c, 0.0)) for c in cats))
            per_column[name] = {"tvd": col_tvd}

        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no categorical columns"],
            )

        mean_tvd = sum(v["tvd"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_tvd)))

        from syneva.render.figures import BarComparison

        worst = max(per_column.items(), key=lambda kv: kv[1]["tvd"])[0]
        r_freq = (
            real[worst].astype("string").fillna("__NA__").value_counts(normalize=True).to_dict()
        )
        s_freq = (
            synthetic[worst]
            .astype("string")
            .fillna("__NA__")
            .value_counts(normalize=True)
            .to_dict()
        )
        payload = BarComparison(real=r_freq, synthetic=s_freq, title=f"TVD: {worst}")
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_tvd": mean_tvd},
            per_column=per_column,
            plot_payload=payload,
            notes=notes,
        )
