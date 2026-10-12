from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import stats

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    # Column-wise coercion turns pd.NA (nullable dtypes) and stray strings into NaN.
    return np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan) for c in cols]
    )


@registry.register
class SlicedWasserstein:
    """Sliced Wasserstein-1 distance between the real and synthetic numeric columns.

    Both datasets are standardized with the real column means and standard
    deviations before projecting, so the distance is in units of real standard
    deviations and does not depend on the columns' scale. Rows with a missing
    or infinite value in any numeric column are dropped (complete-case
    analysis), and columns that are constant in the real data are left out,
    since they have no spread to standardize by.

    References
    ----------
    Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2011). Wasserstein barycenter
    and its application to texture mixing. In *Scale Space and Variational
    Methods in Computer Vision* (LNCS 6667, pp. 435-446). Springer.

    Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and Radon
    Wasserstein barycenters of measures. *Journal of Mathematical Imaging and
    Vision*, 51(1), 22-45.
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="sliced_wasserstein",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nums = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]
        if not nums:
            return MetricResult(
                spec=self.spec,
                skip_reason="There are no numeric columns to compare.",
            )
        X_real = _numeric_matrix(real, nums)
        X_syn = _numeric_matrix(synthetic, nums)
        real_ok = np.isfinite(X_real).all(axis=1)
        syn_ok = np.isfinite(X_syn).all(axis=1)
        notes: list[str] = []
        n_drop_real, n_drop_syn = int((~real_ok).sum()), int((~syn_ok).sum())
        if n_drop_real or n_drop_syn:
            notes.append(
                f"dropped {n_drop_real} of {len(X_real)} real "
                f"({n_drop_real / len(X_real):.1%}) and {n_drop_syn} of "
                f"{len(X_syn)} synthetic ({n_drop_syn / len(X_syn):.1%}) rows "
                "with a missing or non-finite numeric value"
            )
        X_real, X_syn = X_real[real_ok], X_syn[syn_ok]
        if len(X_real) < 2 or len(X_syn) == 0:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "Too few rows without missing or non-finite numeric values: need at "
                    "least 2 real rows and 1 synthetic row."
                ),
            )
        # Exact test: std of a constant float column can be ~1e-17 rather than 0.
        varying = X_real.max(axis=0) > X_real.min(axis=0)
        notes.extend(
            f"column '{name}' skipped (constant in the real data)"
            for name, keep in zip(nums, varying, strict=True)
            if not keep
        )
        if not varying.any():
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason="Every numeric column is constant in the real data.",
            )
        X_real, X_syn = X_real[:, varying], X_syn[:, varying]
        mean = X_real.mean(axis=0)
        std = X_real.std(axis=0, ddof=1)
        X_real = (X_real - mean) / std
        X_syn = (X_syn - mean) / std
        rng = np.random.default_rng(42)
        d = X_real.shape[1]
        n_proj = 50
        dirs = rng.normal(size=(n_proj, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        dists = [stats.wasserstein_distance(X_real @ v, X_syn @ v) for v in dirs]
        sw = float(np.mean(dists))
        score = float(max(0.0, 1.0 - sw / 3.0))
        return MetricResult(
            spec=self.spec, scalars={"score": score, "sliced_wasserstein": sw}, notes=notes
        )
