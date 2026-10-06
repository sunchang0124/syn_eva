from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.distance import gower_matrix
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import _EPS, runnable_specs
from syneva.preservation.spec import SubgroupSpec


@registry.register
@dataclass
class MinorityPrivacyRisk:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_privacy_risk",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    distance: str = "euclidean"
    min_rows: int = 10
    cap: int = 2000
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        if not specs:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "No subgroup spec could be evaluated. Pass subgroup_specs to measure minority "
                    "privacy risk."
                ),
            )
        rng = np.random.default_rng(self.random_state)

        syn = synthetic
        if len(syn) > self.cap:
            syn = syn.iloc[rng.choice(len(syn), self.cap, replace=False)]
            notes.append(f"synthetic capped at {self.cap} rows")

        member_mask = np.zeros(len(real), dtype=bool)
        for sp in specs:
            member_mask |= sp.matches(real).to_numpy()
        member_idx = np.flatnonzero(member_mask)
        other_idx = np.flatnonzero(~member_mask)
        not_measurable = False
        if len(real) > self.cap or len(member_idx) >= self.cap:
            member_budget = self.cap // 2
            if len(member_idx) > member_budget:
                kept_members = rng.choice(member_idx, member_budget, replace=False)
            else:
                kept_members = member_idx
            fill_budget = self.cap - len(kept_members)
            if len(other_idx) > fill_budget:
                kept_others = rng.choice(other_idx, fill_budget, replace=False)
            else:
                kept_others = other_idx
            if len(kept_others) == 0:
                not_measurable = True
                kept_idx = np.arange(len(real))
                for sp in specs:
                    notes.append(
                        f"subgroup '{sp.name}' risk not measurable (subgroup spans all kept rows)"
                    )
            else:
                kept_idx = np.concatenate([kept_members, kept_others])
                notes.append(
                    f"real rows capped at {self.cap} "
                    f"({len(kept_members)} subgroup members + {len(kept_others)} others kept)"
                )
        else:
            kept_idx = np.arange(len(real))
        real_kept = real.iloc[kept_idx].reset_index(drop=True)

        if not_measurable:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "No subgroup risk is measurable: each subgroup spans all kept real rows, so "
                    "there is no baseline to compare against."
                ),
            )

        if self.distance == "gower":
            dcr = gower_matrix(real_kept, syn, meta).min(axis=1)
        else:
            x_real, x_syn = encode_pair(real_kept, syn, meta)
            nn = NearestNeighbors(n_neighbors=1).fit(x_syn)
            dcr = nn.kneighbors(x_real)[0][:, 0]

        overall_med = float(np.median(dcr))
        per_column: dict[str, dict[str, float]] = {}
        ratios: list[float] = []
        for sp in specs:
            smask = sp.matches(real_kept).to_numpy()
            if int(smask.sum()) < self.min_rows:
                notes.append(f"subgroup '{sp.name}' skipped (<{self.min_rows} rows)")
                continue
            sub_med = float(np.median(dcr[smask]))
            if overall_med <= _EPS:
                ratio = 1.0
                notes.append(
                    f"subgroup '{sp.name}': overall distances ~0 (global copying); "
                    "no concentration measurable"
                )
            else:
                ratio = sub_med / (overall_med + _EPS)
            ratios.append(ratio)
            per_column[sp.name] = {
                "risk_ratio": ratio,
                "median_dcr_subgroup": sub_med,
                "median_dcr_overall": overall_med,
            }
        if not ratios:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No subgroup spec could be evaluated.",
            )
        scores = [1.0 if r >= 1.0 else float(max(0.0, r)) for r in ratios]
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": min(scores),
                "mean_risk_ratio": sum(ratios) / len(ratios),
                "worst_risk_ratio": min(ratios),
            },
            per_column=per_column,
            notes=notes,
        )
