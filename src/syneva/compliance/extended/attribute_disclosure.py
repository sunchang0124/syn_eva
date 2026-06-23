from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.neighbors import Neighbors
from syneva.core.registry import registry

# encode_pair only encodes NUMERIC and CATEGORICAL; booleans are not usable QIs here.
_USABLE = (ColumnType.NUMERIC, ColumnType.CATEGORICAL)


@registry.register
class AttributeDisclosure:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="attribute_disclosure",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        sensitive = [n for n, cm in meta.columns.items() if cm.sensitive]
        qi = [n for n, cm in meta.columns.items() if not cm.sensitive and cm.dtype in _USABLE]
        if not sensitive or not qi:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no sensitive columns or no quasi-identifiers; skipped"],
            )
        qi_meta = Metadata(columns={n: meta.columns[n] for n in qi})
        nb = Neighbors(real, synthetic, qi_meta, distance=self.distance, cap=10**9)
        if nb.n_features == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no encodable quasi-identifiers or empty synthetic; skipped"],
            )
        idx = nb.real_to_syn_index()
        real_r = real.reset_index(drop=True)
        syn_r = synthetic.reset_index(drop=True)
        matches = np.ones(len(real_r), dtype=bool)
        for n in sensitive:
            guessed = syn_r[n].to_numpy()[idx]
            actual = real_r[n].to_numpy()
            if meta.columns[n].dtype is ColumnType.NUMERIC:
                col = pd.to_numeric(real_r[n], errors="coerce")
                thr = float(col.max() - col.min()) / 30.0
                g = pd.to_numeric(pd.Series(guessed), errors="coerce").to_numpy()
                a = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy()
                col_match = (np.abs(g - a) <= thr) if thr > 0 else (g == a)
            else:
                g = pd.Series(guessed).astype("string").fillna("__NA__").to_numpy()
                a = pd.Series(actual).astype("string").fillna("__NA__").to_numpy()
                col_match = g == a
            matches &= col_match
        disclosure_rate = float(np.mean(matches))
        score = float(min(1.0, max(0.0, 1.0 - disclosure_rate)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "disclosure_rate": disclosure_rate},
        )
