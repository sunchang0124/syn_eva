from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns


@registry.register
@dataclass
class MinorityClassDensity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_class_density",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        ratios: list[float] = []
        cat_cols = categorical_columns(meta)
        if not cat_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no categorical columns"]
            )
        for name in cat_cols:
            p_real = real[name].value_counts(normalize=True)
            if p_real.empty:
                continue
            # least-frequent class; ties broken by sorted label for determinism
            minority = min(p_real.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
            pr = float(p_real[minority])
            ps = float(synthetic[name].value_counts(normalize=True).get(minority, 0.0))
            ratio = 0.0 if max(pr, ps) == 0 else min(pr, ps) / max(pr, ps)
            ratios.append(ratio)
            per_column[name] = {"density_ratio": ratio}
        if not ratios:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no usable categorical columns"]
            )
        score = float(min(1.0, max(0.0, sum(ratios) / len(ratios))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "worst_density_ratio": min(ratios)},
            per_column=per_column,
        )
