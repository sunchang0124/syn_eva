from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_frames, encode_pair
from syneva.core.distance import gower_matrix, numeric_ranges
from syneva.core.metadata import ColumnType, Metadata

_VALID = ("euclidean", "gower")


def _usable_feature_count(meta: Metadata) -> int:
    return sum(
        1
        for cm in meta.columns.values()
        if cm.dtype in (ColumnType.NUMERIC, ColumnType.CATEGORICAL)
    )


class Neighbors:
    """Nearest-neighbour distance primitives for a (real, synthetic) pair, under
    a euclidean (encode_pair + sklearn) or gower (matrices built on first use) backend."""

    def __init__(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        meta: Metadata,
        *,
        distance: str = "euclidean",
        cap: int = 2000,
        random_state: int = 42,
        holdout: pd.DataFrame | None = None,
    ) -> None:
        if distance not in _VALID:
            raise ValueError(f"unknown distance '{distance}'; use one of {_VALID}")
        self.distance = distance
        self.capped = False
        if distance == "gower":
            # One scale for every matrix: ranges over the full real data, before capping.
            ranges = numeric_ranges(real, meta)
            rng = np.random.default_rng(random_state)
            r, s = real, synthetic
            h = holdout
            if len(r) > cap:
                r = r.iloc[rng.choice(len(r), cap, replace=False)]
                self.capped = True
            if len(s) > cap:
                s = s.iloc[rng.choice(len(s), cap, replace=False)]
                self.capped = True
            if h is not None and len(h) > cap:
                h = h.iloc[rng.choice(len(h), cap, replace=False)]
                self.capped = True
            # Rows are fixed here; each matrix is built on first use (see _gower).
            self._frames = {"real": r, "syn": s, "holdout": h}
            self._meta, self._ranges = meta, ranges
            self._gower_cache: dict[tuple[str, str], np.ndarray] = {}
            self._n_features = _usable_feature_count(meta)
            self._nr, self._ns = len(r), len(s)
        else:
            if holdout is not None:
                self._x_real, self._x_syn, self._x_holdout = encode_frames(
                    [real, synthetic, holdout], meta
                )
            else:
                self._x_real, self._x_syn = encode_pair(real, synthetic, meta)
                self._x_holdout = None
            self._n_features = self._x_real.shape[1]
            self._nr, self._ns = len(self._x_real), len(self._x_syn)
        self._has_holdout = holdout is not None

    @property
    def n_features(self) -> int:
        return self._n_features

    def _gower(self, a: str, b: str) -> np.ndarray:
        """Gower matrix between two of the (capped) frames, built once and reused.
        Callers that modify it must work on a copy."""
        key = (a, b)
        if key not in self._gower_cache:
            self._gower_cache[key] = gower_matrix(
                self._frames[a], self._frames[b], self._meta, self._ranges
            )
        return self._gower_cache[key]

    def _self_dist(self, which: str, k: int) -> np.ndarray:
        if self.distance == "gower":
            d = self._gower(which, which).copy()
            n = d.shape[0]
            if k >= n:
                raise ValueError(
                    f"k={k} too large: only {n - 1} other rows available for self-distance"
                )
            np.fill_diagonal(d, np.inf)
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        x = self._x_real if which == "real" else self._x_syn
        return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[0][:, k]

    def real_self(self, k: int = 1) -> np.ndarray:
        return self._self_dist("real", k)

    def syn_self(self, k: int = 1) -> np.ndarray:
        return self._self_dist("syn", k)

    def _cross_dist(self, src: str, k: int) -> np.ndarray:
        if self.distance == "gower":
            d = self._gower("real", "syn") if src == "real" else self._gower("real", "syn").T
            if k > d.shape[1]:
                raise ValueError(f"k={k} too large: only {d.shape[1]} target rows available")
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        if src == "real":
            return (
                NearestNeighbors(n_neighbors=k)
                .fit(self._x_syn)
                .kneighbors(self._x_real)[0][:, k - 1]
            )
        return (
            NearestNeighbors(n_neighbors=k).fit(self._x_real).kneighbors(self._x_syn)[0][:, k - 1]
        )

    def real_to_syn(self, k: int = 1) -> np.ndarray:
        return self._cross_dist("real", k)

    def syn_to_real(self, k: int = 1) -> np.ndarray:
        return self._cross_dist("syn", k)

    def real_to_syn_index(self) -> np.ndarray:
        if self.distance == "gower":
            return np.argmin(self._gower("real", "syn"), axis=1)
        return (
            NearestNeighbors(n_neighbors=1)
            .fit(self._x_syn)
            .kneighbors(self._x_real, return_distance=False)[:, 0]
        )

    def holdout_to_real(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        if self.distance == "gower":
            d = self._gower("holdout", "real")
            if k > d.shape[1]:
                raise ValueError(f"k={k} too large: only {d.shape[1]} real rows")
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        n_real = len(self._x_real)
        if k > n_real:
            raise ValueError(f"k={k} too large: only {n_real} real rows")
        return (
            NearestNeighbors(n_neighbors=k)
            .fit(self._x_real)
            .kneighbors(self._x_holdout)[0][:, k - 1]
        )

    def holdout_self(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        if self.distance == "gower":
            d = self._gower("holdout", "holdout").copy()
            n = d.shape[0]
            if k >= n:
                raise ValueError(f"k={k} too large: only {n - 1} other holdout rows")
            np.fill_diagonal(d, np.inf)
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        assert self._x_holdout is not None
        n = len(self._x_holdout)
        if k >= n:
            raise ValueError(f"k={k} too large: only {n - 1} other holdout rows")
        x = self._x_holdout
        return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[0][:, k]
