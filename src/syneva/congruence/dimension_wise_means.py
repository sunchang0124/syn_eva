from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.congruence._constant import ConstantRealColumns
from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class DimensionWiseMeans:
    """Absolute difference in column means, in real standard deviations, per numeric column.

    For each numeric column, the difference between the real and synthetic
    means is divided by the real sample standard deviation. The score is
    ``1 -`` the mean over columns, clipped to [0, 1]. This adapts the
    dimension-wise probability comparison of Choi et al. (2017), defined there
    for binary features, to standardised differences in means.

    Columns with fewer than 2 values, and columns that are constant in the real
    data (no spread to standardise by), are left out with a note.

    References
    ----------
    Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017).
    Generating multi-label discrete patient records using generative adversarial
    networks. In *Proceedings of the 2nd Machine Learning for Healthcare
    Conference* (PMLR 68, pp. 286-305).

    Lautrup, A. D., Hyrup, T., Zimek, A., & Schneider-Kamp, P. (2024). SynthEval:
    A framework for detailed utility and privacy evaluation of tabular synthetic
    data. *Data Mining and Knowledge Discovery*.
    https://doi.org/10.1007/s10618-024-01081-4
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="dimension_wise_means",
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
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) < 2 or len(s) < 2:
                notes.append(f"column '{name}' skipped (need >=2 values for std)")
                continue
            if constant.skip(name, r):
                continue
            d = abs(float(r.mean()) - float(s.mean())) / float(r.std())
            per_column[name] = {"std_mean_diff": d}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason=constant.skip_reason("No numeric column has enough values to compare."),
            )
        mean_d = sum(v["std_mean_diff"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_d)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_abs_std_diff": float(mean_d)},
            per_column=per_column,
            notes=notes,
        )
