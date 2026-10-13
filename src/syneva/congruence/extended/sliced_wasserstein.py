from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from scipy import stats

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_NO_USABLE_COLUMN = (
    "No numeric column can be compared: each one is missing in the real or synthetic "
    "data, or constant in the real data."
)


def _varies(x: np.ndarray) -> bool:
    # Exact test: the std of a constant float column can be ~1e-17 rather than 0.
    return bool(x.max() > x.min())


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
    deviations and does not depend on the columns' scale. The scale comes from
    the real data alone, not from both datasets combined, so the synthetic data
    cannot move the yardstick it is measured against.

    Columns with no finite values in the real or the synthetic data, and
    columns that are constant in the real data (no spread to standardize by),
    are left out with a note. Rows with a missing or infinite value in a
    remaining column are then dropped (complete-case analysis). A column that
    is constant once those rows are gone is left out as well, and the rows are
    recomputed without it.

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
        if len(X_real) == 0 or len(X_syn) == 0:
            return MetricResult(
                spec=self.spec, skip_reason="The real or synthetic dataset is empty."
            )
        notes: list[str] = []
        # Choose columns before dropping rows, so an unusable column (all missing, or
        # constant in the real data) does not throw away rows through its missing values.
        keep = np.ones(len(nums), dtype=bool)
        for j, name in enumerate(nums):
            real_vals = X_real[:, j][np.isfinite(X_real[:, j])]
            if real_vals.size == 0 or not np.isfinite(X_syn[:, j]).any():
                notes.append(
                    f"column '{name}' skipped (no finite values in the real or synthetic data)"
                )
                keep[j] = False
            elif not _varies(real_vals):
                notes.append(f"column '{name}' skipped (constant in the real data)")
                keep[j] = False
        # Drop incomplete rows over the kept columns. If that leaves a kept column with a
        # single real value, exclude it too and recompute the rows from the full data, so
        # rows lost only to that column come back. keep only shrinks, so this terminates.
        while True:
            if not keep.any():
                return MetricResult(spec=self.spec, notes=notes, skip_reason=_NO_USABLE_COLUMN)
            real_ok = np.isfinite(X_real[:, keep]).all(axis=1)
            syn_ok = np.isfinite(X_syn[:, keep]).all(axis=1)
            if real_ok.sum() < 2:
                break
            newly_constant = [j for j in np.flatnonzero(keep) if not _varies(X_real[real_ok, j])]
            if not newly_constant:
                break
            for j in newly_constant:
                notes.append(
                    f"column '{nums[j]}' skipped (constant once rows with missing values "
                    "are dropped)"
                )
                keep[j] = False
        n_drop_real, n_drop_syn = int((~real_ok).sum()), int((~syn_ok).sum())
        if n_drop_real or n_drop_syn:
            notes.append(
                f"dropped {n_drop_real} of {len(X_real)} real "
                f"({n_drop_real / len(X_real):.1%}) and {n_drop_syn} of "
                f"{len(X_syn)} synthetic ({n_drop_syn / len(X_syn):.1%}) rows "
                "with a missing or non-finite numeric value"
            )
        X_real, X_syn = X_real[real_ok][:, keep], X_syn[syn_ok][:, keep]
        if len(X_real) < 2 or len(X_syn) == 0:
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "Too few rows without missing or non-finite numeric values: need at "
                    "least 2 real rows and 1 synthetic row."
                ),
            )
        rng = np.random.default_rng(42)
        d = X_real.shape[1]
        n_proj = 50
        dirs = rng.normal(size=(n_proj, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # Dividing by the real max |x| first leaves the standardized values unchanged, but
        # keeps the mean and std in floating-point range for very large or small values.
        # Synthetic values far outside the real range can still overflow; the finiteness
        # check below turns that into a skip instead of a warning and a silent score.
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.abs(X_real).max(axis=0)
            X_real, X_syn = X_real / scale, X_syn / scale
            mean = X_real.mean(axis=0)
            std = X_real.std(axis=0, ddof=1)
            X_real = (X_real - mean) / std
            X_syn = (X_syn - mean) / std
            dists = [stats.wasserstein_distance(X_real @ v, X_syn @ v) for v in dirs]
        sw = float(np.mean(dists))
        if not np.isfinite(sw):
            return MetricResult(
                spec=self.spec,
                notes=notes,
                skip_reason=(
                    "The distance is not finite: the synthetic values are too far outside "
                    "the real range to standardize."
                ),
            )
        score = float(max(0.0, 1.0 - sw / 3.0))
        return MetricResult(
            spec=self.spec, scalars={"score": score, "sliced_wasserstein": sw}, notes=notes
        )
