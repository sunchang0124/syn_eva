from __future__ import annotations

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata


def gower_matrix(a: pd.DataFrame, b: pd.DataFrame, meta: Metadata) -> np.ndarray:
    """Gower distance between every row of `a` and every row of `b`, in [0, 1].

    Numeric features contribute |a-b| / range (range over union(a,b); 0 if range
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
            combined = np.concatenate([av, bv])
            finite = combined[np.isfinite(combined)]
            if finite.size == 0:
                continue
            rng = float(finite.max() - finite.min())
            diff = np.abs(av[:, None] - bv[None, :])
            d = diff / rng if rng > 0 else np.zeros((na, nb))
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
