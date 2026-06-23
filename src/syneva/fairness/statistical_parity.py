from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.fairness.spec import FairnessSpec


def _spd(df: pd.DataFrame, protected: str, outcome: str, favorable: object) -> float:
    sub = df[[protected, outcome]].dropna()
    rates = []
    for g in sub[protected].unique():
        gy = sub[sub[protected] == g][outcome]
        if len(gy) == 0:
            continue
        rates.append(float((gy == favorable).mean()))
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


@registry.register
@dataclass
class StatisticalParity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="statistical_parity",
        c="fairness",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    specs: list[FairnessSpec] = field(default_factory=list)
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        drifts: list[float] = []
        headline: tuple[float, float, float] | None = None
        for sp in self.specs:
            cols = {sp.protected_attribute, sp.outcome}
            if not cols <= set(real.columns) or not cols <= set(synthetic.columns):
                notes.append(
                    f"spec {sp.protected_attribute}->{sp.outcome} skipped (missing column)"
                )
                continue
            favorable = sp.favorable_outcome
            if favorable is None:
                favorable = max(
                    pd.concat([real[sp.outcome], synthetic[sp.outcome]]).dropna().unique()
                )
            spd_r = _spd(real, sp.protected_attribute, sp.outcome, favorable)
            spd_s = _spd(synthetic, sp.protected_attribute, sp.outcome, favorable)
            drift = abs(spd_s - spd_r)
            per_column[f"{sp.protected_attribute}|{sp.outcome}"] = {
                "spd_synthetic": spd_s,
                "spd_real": spd_r,
                "parity_drift": drift,
            }
            drifts.append(drift)
            if headline is None:
                headline = (spd_s, spd_r, drift)
        if not drifts:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable fairness specs"],
            )
        mean_drift = sum(drifts) / len(drifts)
        score = float(min(1.0, max(0.0, 1.0 - mean_drift)))
        spd_s, spd_r, drift = headline  # type: ignore[misc]
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "spd_synthetic": spd_s,
                "spd_real": spd_r,
                "parity_drift": drift,
            },
            per_column=per_column,
            notes=notes,
        )
