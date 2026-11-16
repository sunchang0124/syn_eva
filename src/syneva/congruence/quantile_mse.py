from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.congruence._constant import ConstantRealColumns
from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_QUANTILES = np.arange(0.05, 1.0, 0.05)


@registry.register
class QuantileMSE:
    """Mean squared difference between real and synthetic quantiles, per numeric column.

    For each numeric column, the 5th to 95th percentiles (in steps of 5) of the
    real and synthetic values are compared, with differences in units of the
    real (population) standard deviation. The score is ``1 - sqrt`` of the mean
    over columns, clipped to [0, 1].

    Columns empty after dropping missing values, and columns that are constant
    in the real data (no spread to standardise by), are left out with a note.

    References
    ----------
    Lautrup, A. D., Hyrup, T., Zimek, A., & Schneider-Kamp, P. (2024). SynthEval:
    A framework for detailed utility and privacy evaluation of tabular synthetic
    data. *Data Mining and Knowledge Discovery*.
    https://doi.org/10.1007/s10618-024-01081-4
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="quantile_mse",
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
        constant = ConstantRealColumns(notes)
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna().to_numpy()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            if constant.skip(name, r):
                continue
            scale = float(np.std(r))
            qr = np.quantile(r, _QUANTILES)
            qs = np.quantile(s, _QUANTILES)
            qmse = float(np.mean(((qr - qs) / scale) ** 2))
            per_column[name] = {"quantile_mse": qmse}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason=constant.skip_reason("No numeric column has values to compare."),
            )
        mean_qmse = sum(v["quantile_mse"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - np.sqrt(mean_qmse))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_quantile_mse": float(mean_qmse)},
            per_column=per_column,
            notes=notes,
        )
