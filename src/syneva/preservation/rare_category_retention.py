from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns


@registry.register
@dataclass
class RareCategoryRetention:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="rare_category_retention",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    rare_threshold: float = 0.05

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        retentions: list[float] = []
        n_missing = 0
        cat_cols = categorical_columns(meta)
        if not cat_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no categorical columns"]
            )
        for name in cat_cols:
            p_real = real[name].value_counts(normalize=True)
            p_real = p_real[p_real > 0]
            p_syn = synthetic[name].value_counts(normalize=True)
            rare = p_real[p_real < self.rare_threshold]
            if rare.empty:
                continue
            col_retentions: list[float] = []
            for cat, pr in rare.items():
                ps = float(p_syn.get(cat, 0.0))
                if ps == 0.0:
                    n_missing += 1
                col_retentions.append(min(ps / float(pr), 1.0))
            retentions.extend(col_retentions)
            per_column[name] = {"retention": sum(col_retentions) / len(col_retentions)}
        if not retentions:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no rare categories in the real data"],
            )
        score = float(min(1.0, max(0.0, sum(retentions) / len(retentions))))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "n_rare_categories": float(len(retentions)),
                "pct_rare_missing": n_missing / len(retentions),
            },
            per_column=per_column,
            notes=notes,
        )
