from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata


def _finite_numeric(values: pd.Series) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def numeric_ranges(df: pd.DataFrame, meta: Metadata) -> dict[str, float]:
    """Range (max - min over finite values) of every numeric column of `df`.

    Pass the full real data, before any capping, so every Gower matrix in one
    evaluation shares the same scale. Columns with no finite values are left out.
    """
    out: dict[str, float] = {}
    for name, cm in meta.columns.items():
        if cm.dtype is ColumnType.NUMERIC:
            finite = _finite_numeric(df[name])
            if finite.size:
                out[name] = float(finite.max() - finite.min())
    return out


def gower_matrix(
    a: pd.DataFrame,
    b: pd.DataFrame,
    meta: Metadata,
    ranges: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Gower distance between every row of `a` and every row of `b`, in [0, 1].

    Gower's coefficient (Gower, J. C. (1971). A general coefficient of similarity
    and some of its properties. Biometrics 27(4), 857-871) divides each numeric
    difference by one fixed range per feature. With `ranges` (see
    `numeric_ranges`), numeric features contribute min(|a-b| / range, 1); a
    zero range contributes 0 if equal else 1; columns not in `ranges` are
    skipped. Without `ranges`, the range is taken over union(a, b) (0 if range
    0). Categorical features contribute 0 if equal else 1 (NaN -> '__NA__').
    The distance is the mean over usable (numeric/categorical) features. NaN
    numeric pairs contribute 0 (no disagreement). No usable features -> zeros.
    """
    na, nb = len(a), len(b)
    total = np.zeros((na, nb), dtype=float)
    count = 0
    for name, cm in meta.columns.items():
        if cm.dtype is ColumnType.NUMERIC:
            av = pd.to_numeric(a[name], errors="coerce").to_numpy(dtype=float)
            bv = pd.to_numeric(b[name], errors="coerce").to_numpy(dtype=float)
            diff = np.abs(av[:, None] - bv[None, :])
            if ranges is None:
                finite = np.concatenate([_finite_numeric(a[name]), _finite_numeric(b[name])])
                if finite.size == 0:
                    continue
                rng = float(finite.max() - finite.min())
                d = diff / rng if rng > 0 else np.zeros((na, nb))
            else:
                if name not in ranges:
                    continue
                rng = ranges[name]
                with np.errstate(invalid="ignore"):
                    d = np.minimum(diff / rng, 1.0) if rng > 0 else (diff > 0).astype(float)
            total += np.where(np.isfinite(d), d, 0.0)
            count += 1
        elif cm.dtype is ColumnType.CATEGORICAL:
            av = a[name].astype("string").fillna("__NA__").to_numpy()
            bv = b[name].astype("string").fillna("__NA__").to_numpy()
            total += (av[:, None] != bv[None, :]).astype(float)
            count += 1
    if count == 0:
        return np.zeros((na, nb))
    return total / count
