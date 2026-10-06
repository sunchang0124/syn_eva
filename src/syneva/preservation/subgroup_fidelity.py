from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from scipy.stats import ks_2samp

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns, numeric_columns, runnable_specs
from syneva.preservation.spec import SubgroupSpec


def _column_distances(real_sub: pd.DataFrame, syn_sub: pd.DataFrame, meta) -> list[float]:
    """Per-column distributional distances within the subgroup, each in [0, 1]:
    two-sample KS statistic for numeric columns, total-variation distance for
    categorical/boolean columns."""
    dists: list[float] = []
    for name in numeric_columns(meta):
        rv = pd.to_numeric(real_sub[name], errors="coerce").dropna()
        sv = pd.to_numeric(syn_sub[name], errors="coerce").dropna()
        if rv.empty or sv.empty:
            continue
        dists.append(float(ks_2samp(rv, sv).statistic))  # pyright: ignore[reportAttributeAccessIssue]  # scipy result types are untyped
    for name in categorical_columns(meta):
        pr = real_sub[name].value_counts(normalize=True)
        ps = syn_sub[name].value_counts(normalize=True)
        cats = set(pr.index) | set(ps.index)
        if not cats:
            continue
        dists.append(0.5 * sum(abs(float(pr.get(c, 0.0)) - float(ps.get(c, 0.0))) for c in cats))
    return dists


@registry.register
@dataclass
class SubgroupFidelity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="subgroup_fidelity",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    min_rows: int = 10

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        per_column: dict[str, dict[str, float]] = {}
        scores: list[float] = []
        share_drifts: list[float] = []
        for sp in specs:
            rmask = sp.matches(real)
            if int(rmask.sum()) < self.min_rows:
                notes.append(f"subgroup '{sp.name}' skipped (<{self.min_rows} real rows)")
                continue
            smask = sp.matches(synthetic)
            share_real = float(rmask.mean())
            share_syn = float(smask.mean())
            share_drifts.append(abs(share_syn - share_real))
            if int(smask.sum()) == 0:
                notes.append(f"subgroup '{sp.name}' erased in the synthetic data")
                per_column[sp.name] = {
                    "share_real": share_real,
                    "share_syn": 0.0,
                    "shape_score": 0.0,
                    "score": 0.0,
                }
                scores.append(0.0)
                continue
            share_score = min(share_syn, share_real) / max(share_syn, share_real)
            dists = _column_distances(real[rmask], synthetic[smask], meta)
            shape_score = 1.0 - (sum(dists) / len(dists)) if dists else 1.0
            score_spec = float(min(1.0, max(0.0, 0.5 * shape_score + 0.5 * share_score)))
            per_column[sp.name] = {
                "share_real": share_real,
                "share_syn": share_syn,
                "shape_score": shape_score,
                "score": score_spec,
            }
            scores.append(score_spec)
        if not scores:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="No subgroup spec could be evaluated.",
            )
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float(sum(scores) / len(scores)),
                "worst_subgroup_score": min(scores),
                "mean_share_drift": sum(share_drifts) / len(share_drifts),
            },
            per_column=per_column,
            notes=notes,
        )
