from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_CAP = 2000


@registry.register
class HittingRate:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="hitting_rate",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        num = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]
        cat = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.CATEGORICAL]
        if not num and not cat:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "hit_rate": 0.0},
                notes=["no usable columns"],
            )
        rng = np.random.default_rng(42)
        notes: list[str] = []
        r, s = real, synthetic
        if len(r) > _CAP:
            r = r.iloc[rng.choice(len(r), _CAP, replace=False)]
            notes.append("real capped at 2000 rows")
        if len(s) > _CAP:
            s = s.iloc[rng.choice(len(s), _CAP, replace=False)]
            notes.append("synthetic capped at 2000 rows")
        thresh: dict[str, float] = {}
        for n in num:
            col = pd.to_numeric(real[n], errors="coerce")
            thresh[n] = float(col.max() - col.min()) / 30.0
        r_num = {n: pd.to_numeric(r[n], errors="coerce").to_numpy() for n in num}
        s_num = {n: pd.to_numeric(s[n], errors="coerce").to_numpy() for n in num}
        r_cat = {n: r[n].astype("string").fillna("__NA__").to_numpy() for n in cat}
        s_cat = {n: s[n].astype("string").fillna("__NA__").to_numpy() for n in cat}
        hits = 0
        for i in range(len(r)):
            mask = np.ones(len(s), dtype=bool)
            for n in num:
                if thresh[n] <= 0:
                    mask &= s_num[n] == r_num[n][i]
                else:
                    mask &= np.abs(s_num[n] - r_num[n][i]) <= thresh[n]
                if not mask.any():
                    break
            if mask.any():
                for n in cat:
                    mask &= s_cat[n] == r_cat[n][i]
                    if not mask.any():
                        break
            if mask.any():
                hits += 1
        hit_rate = float(hits / len(r)) if len(r) else 0.0
        score = float(min(1.0, max(0.0, 1.0 - hit_rate)))
        return MetricResult(
            spec=self.spec, scalars={"score": score, "hit_rate": hit_rate}, notes=notes
        )
