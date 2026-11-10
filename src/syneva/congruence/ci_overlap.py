from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12


@registry.register
class CIOverlap:
    """Overlap of the 95% confidence intervals of each numeric column's mean.

    Uses the interval overlap measure of Karr et al. (2006),
    ``0.5 * (overlap / width_real + overlap / width_synthetic)``, clamped to [0, 1].

    By default each interval uses its own dataset's standard error, so a larger
    synthetic set gives a narrower interval and, when nested, a score nearer 0.5.
    ``use_real_se=True`` instead builds both intervals from the real data's standard
    error. This targets the estimate the real data would give rather than the
    population value (Raab et al., 2017), with the synthetic precision taken at the
    real sample size. Both intervals then have the same length, and Karr's measure
    reduces to ``1 - |z| / (2 * 1.96)``, where ``z`` is the difference in means divided
    by the real data's standard error: the standardised difference of Snoke et al.
    (2018). The score no longer depends on the synthetic sample size.

    References
    ----------
    Karr, A. F., Kohnen, C. N., Oganian, A., Reiter, J. P., & Sanil, A. P. (2006).
    A framework for evaluating the utility of data altered to protect confidentiality.
    The American Statistician, 60(3), 224-232.

    Raab, G. M., Nowok, B., & Dibben, C. (2017). Practical data synthesis for large
    samples. Journal of Privacy and Confidentiality, 7(3), 67-97.

    Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavkovic, A. (2018). General and
    specific utility measures for synthetic data. Journal of the Royal Statistical
    Society: Series A, 181(3), 663-688.
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="ci_overlap",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def __init__(self, use_real_se: bool = False) -> None:
        self.use_real_se = use_real_se

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) < 2 or len(s) < 2:
                notes.append(f"column '{name}' skipped (need >=2 values)")
                continue
            se_r = float(r.std()) / math.sqrt(len(r))
            se_s = se_r if self.use_real_se else float(s.std()) / math.sqrt(len(s))
            lo_r, hi_r = float(r.mean()) - 1.96 * se_r, float(r.mean()) + 1.96 * se_r
            lo_s, hi_s = float(s.mean()) - 1.96 * se_s, float(s.mean()) + 1.96 * se_s
            w_r, w_s = hi_r - lo_r, hi_s - lo_s
            if w_r < _EPS and w_s < _EPS:
                # Zero-width CIs (constant column): full overlap iff the means match.
                frac = 1.0 if abs(float(r.mean()) - float(s.mean())) < _EPS else 0.0
            elif w_r < _EPS or w_s < _EPS:
                # A point shares no length with an interval.
                frac = 0.0
            else:
                # Karr et al. (2006): average the overlap's share of each interval, so a
                # narrow CI nested inside a wide one is not judged by the average width.
                overlap = max(0.0, min(hi_r, hi_s) - max(lo_r, lo_s))
                frac = float(min(1.0, 0.5 * (overlap / w_r + overlap / w_s)))
            per_column[name] = {"ci_overlap": frac}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                per_column={},
                notes=notes,
                skip_reason="No numeric column has enough values to compute a confidence interval.",
            )
        mean_overlap = sum(v["ci_overlap"] for v in per_column.values()) / len(per_column)
        return MetricResult(
            spec=self.spec,
            scalars={"score": float(mean_overlap), "mean_ci_overlap": float(mean_overlap)},
            per_column=per_column,
            notes=notes,
        )
