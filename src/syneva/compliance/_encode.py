from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from syneva.core.metadata import ColumnType, Metadata


def encode_pair(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    meta: Metadata,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode real and synthetic into a shared numeric space for distance metrics.

    Real and synthetic are encoded *together* so the two matrices live in the
    same space: numeric columns are standardized with one scaler fit on the
    combined data, and categorical columns are one-hot encoded over the union
    of categories. Returns ``(X_real, X_syn)``.
    """
    n_real = len(real)
    combined = pd.concat([real, synthetic], ignore_index=True)
    pieces: list[np.ndarray] = []
    for name, cmeta in meta.columns.items():
        if cmeta.dtype is ColumnType.NUMERIC:
            v = pd.to_numeric(combined[name], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
            pieces.append(StandardScaler().fit_transform(v))
        elif cmeta.dtype is ColumnType.CATEGORICAL:
            dummies = pd.get_dummies(combined[name].astype("string").fillna("__NA__"), dtype=float)
            pieces.append(dummies.to_numpy())
    if not pieces:
        empty = np.zeros((len(combined), 0))
        return empty[:n_real], empty[n_real:]
    X = np.concatenate(pieces, axis=1)
    return X[:n_real], X[n_real:]
