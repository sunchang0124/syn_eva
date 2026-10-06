from __future__ import annotations

import math
from itertools import combinations
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _cramers_v(s1: pd.Series, s2: pd.Series) -> float | None:
    """Cramér's V for two categorical series."""
    ct = pd.crosstab(s1, s2)
    if ct.size == 0:
        return None
    r, k = ct.shape
    if min(r, k) <= 1:
        # Only one level in at least one variable — association undefined.
        return None
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = int(ct.values.sum())
    denom = n * (min(r, k) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0  # pyright: ignore[reportOperatorIssue]  # scipy result types are untyped


def _eta_squared(num: pd.Series, cat: pd.Series) -> float | None:
    """Correlation ratio η² (numeric explained by categorical)."""
    grand_mean = num.mean()
    ss_total = float(((num - grand_mean) ** 2).sum())
    if ss_total == 0:
        # Constant numeric column — undefined.
        return None
    groups = num.groupby(cat).agg(["mean", "count"])
    ss_between = float(((groups["mean"] - grand_mean) ** 2 * groups["count"]).sum())
    return ss_between / ss_total


def _pair_assoc(df: pd.DataFrame, a: str, b: str, ta: ColumnType, tb: ColumnType) -> float | None:
    """Return association measure for a column pair in one dataframe."""
    if ta is ColumnType.NUMERIC and tb is ColumnType.NUMERIC:
        # Constant column -> NaN (handled below); silence numpy's divide-by-zero warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(df[a].corr(df[b], method="pearson"))
        return None if np.isnan(corr) else corr
    if ta is ColumnType.CATEGORICAL and tb is ColumnType.CATEGORICAL:
        return _cramers_v(df[a], df[b])
    if {ta, tb} == {ColumnType.NUMERIC, ColumnType.CATEGORICAL}:
        num_col = a if ta is ColumnType.NUMERIC else b
        cat_col = b if ta is ColumnType.NUMERIC else a
        return _eta_squared(df[num_col], df[cat_col])
    return None


@registry.register
class CorrelationDifference:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="correlation_difference",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="pairwise",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        notes: list[str] = []

        eligible = [
            (n, cm.dtype)
            for n, cm in meta.columns.items()
            if cm.dtype in (ColumnType.NUMERIC, ColumnType.CATEGORICAL)
        ]

        per_column: dict[str, dict[str, float]] = {}
        for (a, ta), (b, tb) in combinations(eligible, 2):
            c_real = _pair_assoc(real, a, b, ta, tb)
            c_syn = _pair_assoc(synthetic, a, b, ta, tb)
            if c_real is None or c_syn is None:
                if c_real is None and c_syn is None:
                    _which = "real and synthetic"
                elif c_real is None:
                    _which = "real"
                else:
                    _which = "synthetic"
                notes.append(
                    f"pair '{a}|{b}' skipped (undefined association in {_which} — "
                    "constant column or single category)"
                )
                continue
            per_column[f"{a}|{b}"] = {"abs_corr_diff": abs(c_real - c_syn)}

        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No column pair has a defined association in both datasets.",
            )

        mean_diff = sum(v["abs_corr_diff"] for v in per_column.values()) / len(per_column)
        score = float(max(0.0, min(1.0, 1.0 - mean_diff)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_abs_corr_diff": mean_diff},
            per_column=per_column,
            notes=notes,
        )
