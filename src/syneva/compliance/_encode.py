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
    x_real, x_syn = encode_frames([real, synthetic], meta)
    return x_real, x_syn


def encode_frames(frames: list[pd.DataFrame], meta: Metadata) -> list[np.ndarray]:
    """Encode several frames into ONE shared numeric space (encode_pair is the
    two-frame case): numeric columns standardized with a single scaler fit on the
    concatenation, categoricals one-hot over the union of categories. Returns one
    matrix per input frame, in order."""
    sizes = [len(f) for f in frames]
    combined = pd.concat(frames, ignore_index=True)
    pieces: list[np.ndarray] = []
    for name, cmeta in meta.columns.items():
        if cmeta.dtype is ColumnType.NUMERIC:
            v = pd.to_numeric(combined[name], errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
            pieces.append(StandardScaler().fit_transform(v))
        elif cmeta.dtype is ColumnType.CATEGORICAL:
            dummies = pd.get_dummies(combined[name].astype("string").fillna("__NA__"), dtype=float)
            pieces.append(dummies.to_numpy())
    x = np.concatenate(pieces, axis=1) if pieces else np.zeros((len(combined), 0))
    out: list[np.ndarray] = []
    start = 0
    for n in sizes:
        out.append(x[start : start + n])
        start += n
    return out
