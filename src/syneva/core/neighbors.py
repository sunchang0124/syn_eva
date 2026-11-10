from __future__ import annotations

from collections.abc import Callable

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
    a euclidean (encode_pair, exact nearest neighbours) or gower (matrices built on
    first use) backend.

    Every distance method caches its result per k and returns it read-only, since
    one instance can be shared by several metrics; a caller that modifies a result
    must work on a copy."""

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
        self._has_holdout = holdout is not None
        # Source sizes, checked by resolve() against the frames a metric is given.
        self._source_rows = (len(real), len(synthetic))
        self._results: dict[tuple[str, int], np.ndarray] = {}
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
            # Rows are fixed here; each matrix is built on first use (see _gower). The
            # frames (at most `cap` rows each) are copied, so a caller changing its own
            # frame afterwards can't change a matrix that isn't built yet.
            self._frames = {
                "real": r.copy(),
                "syn": s.copy(),
                "holdout": None if h is None else h.copy(),
            }
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

    @classmethod
    def resolve(
        cls,
        injected: Neighbors | None,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        meta: Metadata,
        *,
        distance: str = "euclidean",
        holdout: pd.DataFrame | None = None,
    ) -> Neighbors:
        """The Neighbors a metric should use: `injected` (shared by the runner) when
        given, else one built for these frames. An injected instance must match what
        the metric would build itself, or its results would differ from a standalone
        run."""
        if injected is None:
            return cls(real, synthetic, meta, distance=distance, holdout=holdout)
        if injected.distance != distance:
            raise ValueError(
                f"injected Neighbors uses '{injected.distance}' distance, "
                f"but the metric uses '{distance}'"
            )
        if injected._source_rows != (len(real), len(synthetic)):
            raise ValueError(
                f"injected Neighbors was built for {injected._source_rows} (real, synthetic) "
                f"rows, but the metric was given {(len(real), len(synthetic))}"
            )
        if holdout is not None and not injected._has_holdout:
            raise ValueError("the metric has a holdout, but the injected Neighbors has none")
        if distance == "euclidean" and holdout is None and injected._has_holdout:
            # The euclidean encoding is fit on the holdout too, so distances would differ.
            # Gower scales by the real data only, so a holdout doesn't change them.
            raise ValueError("the metric has no holdout, but the injected Neighbors has one")
        return injected

    def _cached(self, name: str, k: int, compute: Callable[[], np.ndarray]) -> np.ndarray:
        key = (name, k)
        if key not in self._results:
            out = compute()
            out.setflags(write=False)
            self._results[key] = out
        return self._results[key]

    @property
    def n_features(self) -> int:
        return self._n_features

    def _gower(self, a: str, b: str) -> np.ndarray:
        """Gower matrix between two of the (capped) frames, built once and reused.
        Read-only, since every metric sharing this instance reads the same array;
        callers that modify it must work on a copy."""
        key = (a, b)
        if key not in self._gower_cache:
            d = gower_matrix(self._frames[a], self._frames[b], self._meta, self._ranges)
            d.setflags(write=False)
            self._gower_cache[key] = d
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
        return self._cached("real_self", k, lambda: self._self_dist("real", k))

    def syn_self(self, k: int = 1) -> np.ndarray:
        return self._cached("syn_self", k, lambda: self._self_dist("syn", k))

    def _cross_dist(self, src: str, k: int) -> np.ndarray:
        if self.distance == "gower":
            d = self._gower("real", "syn")
            if src != "real":
                d = d.T
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
        return self._cached("real_to_syn", k, lambda: self._cross_dist("real", k))

    def syn_to_real(self, k: int = 1) -> np.ndarray:
        return self._cached("syn_to_real", k, lambda: self._cross_dist("syn", k))

    def real_to_syn_index(self) -> np.ndarray:
        return self._cached("real_to_syn_index", 1, self._real_to_syn_index)

    def _real_to_syn_index(self) -> np.ndarray:
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
        return self._cached("holdout_to_real", k, lambda: self._holdout_to_real(k))

    def _holdout_to_real(self, k: int) -> np.ndarray:
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

    def holdout_to_syn(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        return self._cached("holdout_to_syn", k, lambda: self._holdout_to_syn(k))

    def _holdout_to_syn(self, k: int) -> np.ndarray:
        if self.distance == "gower":
            d = self._gower("holdout", "syn")
            if k > d.shape[1]:
                raise ValueError(f"k={k} too large: only {d.shape[1]} synthetic rows")
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        if k > self._ns:
            raise ValueError(f"k={k} too large: only {self._ns} synthetic rows")
        return (
            NearestNeighbors(n_neighbors=k)
            .fit(self._x_syn)
            .kneighbors(self._x_holdout)[0][:, k - 1]
        )

    def holdout_self(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        return self._cached("holdout_self", k, lambda: self._holdout_self(k))

    def _holdout_self(self, k: int) -> np.ndarray:
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
