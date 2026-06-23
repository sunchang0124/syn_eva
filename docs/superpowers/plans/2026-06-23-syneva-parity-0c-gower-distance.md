# Gower Distance Backend (Phase 0c-a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Gower distance backend (`distance="gower"`) to syneva's 7 nearest-neighbour metrics via a shared `Neighbors` helper, with euclidean (the default) preserved byte-for-byte.

**Architecture:** A `core/distance.py:gower_matrix()` computes mixed-type Gower distances. A `core/neighbors.py:Neighbors` helper exposes the neighbour primitives the metrics need, with a euclidean backend (`encode_pair` + sklearn, identical to today) and a gower backend (precomputed matrices, capped). The 7 NN metrics are refactored to call `Neighbors` and gain a `distance` field; `evaluate()` threads `distance` through `_instantiate`.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-23-syneva-parity-0c-gower-distance.md`.

---

## CRITICAL environment notes

- Use `uv`. Targeted tests MUST use `--no-cov` (repo `addopts` has `--cov-fail-under=85`, which fails subset runs). Only the full `uv run pytest` enforces the gate.
- Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, then commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Hard constraint:** the euclidean default must be behaviour-preserving. After each metric refactor, the metric's EXISTING unit test must pass UNCHANGED, and the golden snapshot must stay valid (do NOT `--update-golden`). If an existing euclidean test changes value, the refactor is wrong — stop and report.

## File structure

- Create: `src/syneva/core/distance.py` (`gower_matrix`)
- Create: `src/syneva/core/neighbors.py` (`Neighbors`)
- Modify: `src/syneva/compliance/dcr.py`, `nndr.py`, `compliance/extended/epsilon_identifiability.py`, `attribute_disclosure.py`, `coverage/extended/nn_adversarial_accuracy.py`, `authenticity.py`, `alpha_precision.py`
- Modify: `src/syneva/core/runner.py` (`distance` arg + `_instantiate`)
- Test: `tests/unit/core/test_distance.py`, `tests/unit/core/test_neighbors.py`, `tests/integration/test_gower_backend.py`, plus a gower smoke test appended to each metric's existing test file.

---

## Task 1: `gower_matrix`

**Files:**
- Create: `src/syneva/core/distance.py`
- Test: `tests/unit/core/test_distance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_distance.py
import numpy as np
import pandas as pd

from syneva.core.distance import gower_matrix
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "num": ColumnMetadata(name="num", dtype=ColumnType.NUMERIC),
            "cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_rows_zero_distance():
    df = pd.DataFrame({"num": [0.0, 5.0, 10.0], "cat": ["x", "y", "x"]})
    d = gower_matrix(df, df, _meta())
    assert np.allclose(np.diag(d), 0.0)


def test_mixed_example_values():
    a = pd.DataFrame({"num": [0.0], "cat": ["x"]})
    b = pd.DataFrame({"num": [10.0, 5.0], "cat": ["y", "x"]})
    # range over union = 10; row0: num 10/10=1, cat 1 -> mean 1.0; row1: num 0.5, cat 0 -> 0.25
    d = gower_matrix(a, b, _meta())
    assert d.shape == (1, 2)
    assert abs(d[0, 0] - 1.0) < 1e-9
    assert abs(d[0, 1] - 0.25) < 1e-9


def test_no_usable_columns_zero():
    df = pd.DataFrame({"id": ["a", "b"]})
    meta = Metadata(columns={"id": ColumnMetadata(name="id", dtype=ColumnType.ID)})
    d = gower_matrix(df, df, meta)
    assert d.shape == (2, 2)
    assert np.allclose(d, 0.0)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/core/test_distance.py --no-cov -v`.

- [ ] **Step 3: Implement**

```python
# src/syneva/core/distance.py
from __future__ import annotations

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata


def gower_matrix(a: pd.DataFrame, b: pd.DataFrame, meta: Metadata) -> np.ndarray:
    """Gower distance between every row of `a` and every row of `b`, in [0, 1].

    Numeric features contribute |a-b| / range (range over a∪b; 0 if the range is
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
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/core/test_distance.py --no-cov -v` (3 passed).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(core): gower_matrix mixed-type distance"
```

---

## Task 2: `Neighbors` helper (euclidean + gower)

**Files:**
- Create: `src/syneva/core/neighbors.py`
- Test: `tests/unit/core/test_neighbors.py`

- [ ] **Step 1: Write the failing test** (euclidean equivalence + gower ordering)

```python
# tests/unit/core/test_neighbors.py
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import Metadata
from syneva.core.neighbors import Neighbors


def _data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)})
    syn = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    return real, syn, Metadata.infer(real)


def test_euclidean_matches_direct_sklearn():
    real, syn, meta = _data()
    nb = Neighbors(real, syn, meta, distance="euclidean")
    x_real, x_syn = encode_pair(real, syn, meta)
    # real_self(1)
    exp_self = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
    assert np.allclose(nb.real_self(1), exp_self)
    # real_to_syn(1)
    exp_rs = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
    assert np.allclose(nb.real_to_syn(1), exp_rs)
    # syn_to_real(1)
    exp_sr = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(x_syn)[0][:, 0]
    assert np.allclose(nb.syn_to_real(1), exp_sr)
    # nearest syn index
    exp_idx = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(
        x_real, return_distance=False
    )[:, 0]
    assert np.array_equal(nb.real_to_syn_index(), exp_idx)
    assert nb.n_features == x_real.shape[1]


def test_gower_self_excludes_self_and_orders():
    real = pd.DataFrame({"n": [0.0, 1.0, 100.0], "c": ["a", "a", "b"]})
    syn = pd.DataFrame({"n": [0.0], "c": ["a"]})
    meta = Metadata.infer(real)
    nb = Neighbors(real, syn, meta, distance="gower")
    ds = nb.real_self(1)
    assert len(ds) == 3
    assert (ds >= 0).all()
    # row 0 and row 1 are close (small n diff, same cat); row 2 is far -> larger self dist
    assert ds[2] > ds[0]


def test_invalid_distance_raises():
    real, syn, meta = _data()
    with pytest.raises(ValueError, match="distance"):
        Neighbors(real, syn, meta, distance="manhattan")
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/core/test_neighbors.py --no-cov -v`.

- [ ] **Step 3: Implement**

```python
# src/syneva/core/neighbors.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.distance import gower_matrix
from syneva.core.metadata import ColumnType, Metadata

_VALID = ("euclidean", "gower")


def _usable_feature_count(meta: Metadata) -> int:
    return sum(
        1 for cm in meta.columns.values() if cm.dtype in (ColumnType.NUMERIC, ColumnType.CATEGORICAL)
    )


class Neighbors:
    """Nearest-neighbour distance primitives for a (real, synthetic) pair, under
    a euclidean (encode_pair + sklearn) or gower (precomputed matrices) backend."""

    def __init__(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        meta: Metadata,
        *,
        distance: str = "euclidean",
        cap: int = 2000,
        random_state: int = 42,
    ) -> None:
        if distance not in _VALID:
            raise ValueError(f"unknown distance '{distance}'; use one of {_VALID}")
        self.distance = distance
        self.capped = False
        if distance == "gower":
            rng = np.random.default_rng(random_state)
            r, s = real, synthetic
            if len(r) > cap:
                r = r.iloc[rng.choice(len(r), cap, replace=False)]
                self.capped = True
            if len(s) > cap:
                s = s.iloc[rng.choice(len(s), cap, replace=False)]
                self.capped = True
            self._d_rr = gower_matrix(r, r, meta)
            self._d_rs = gower_matrix(r, s, meta)
            self._d_ss = gower_matrix(s, s, meta)
            self._n_features = _usable_feature_count(meta)
            self._nr, self._ns = len(r), len(s)
        else:
            self._x_real, self._x_syn = encode_pair(real, synthetic, meta)
            self._n_features = self._x_real.shape[1]
            self._nr, self._ns = len(self._x_real), len(self._x_syn)

    @property
    def n_features(self) -> int:
        return self._n_features

    def _self_dist(self, which: str, k: int) -> np.ndarray:
        if self.distance == "gower":
            d = (self._d_rr if which == "real" else self._d_ss).copy()
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
            d = self._d_rs if src == "real" else self._d_rs.T
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        if src == "real":
            return NearestNeighbors(n_neighbors=k).fit(self._x_syn).kneighbors(self._x_real)[0][:, k - 1]
        return NearestNeighbors(n_neighbors=k).fit(self._x_real).kneighbors(self._x_syn)[0][:, k - 1]

    def real_to_syn(self, k: int = 1) -> np.ndarray:
        return self._cross_dist("real", k)

    def syn_to_real(self, k: int = 1) -> np.ndarray:
        return self._cross_dist("syn", k)

    def real_to_syn_index(self) -> np.ndarray:
        if self.distance == "gower":
            return np.argmin(self._d_rs, axis=1)
        return (
            NearestNeighbors(n_neighbors=1)
            .fit(self._x_syn)
            .kneighbors(self._x_real, return_distance=False)[:, 0]
        )
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/core/test_neighbors.py --no-cov -v` (3 passed).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(core): Neighbors helper with euclidean + gower backends"
```

---

## Tasks 3–9: refactor the 7 NN metrics to use `Neighbors`

For EACH metric below, the pattern is identical:
1. Add `def __init__(self, distance: str = "euclidean") -> None: self.distance = distance`.
2. Replace the `encode_pair(...)` + `NearestNeighbors(...)` block with `nb = Neighbors(real, synthetic, meta, distance=self.distance)` and the `nb.*` calls shown.
3. Replace the old `X_real.shape[1] == 0`-style guard with `nb.n_features == 0`.
4. Drop the now-unused imports (`encode_pair`, `NearestNeighbors`); keep `numpy`/`pandas` as needed; add `from syneva.core.neighbors import Neighbors`.
5. Append a gower smoke test to the metric's existing test file.
6. **Run the metric's EXISTING test unchanged — it MUST pass** (euclidean preserved), plus the new gower test.

### Task 3: `dcr`

**Files:** Modify `src/syneva/compliance/dcr.py`; Test append `tests/unit/compliance/test_dcr.py`.

- [ ] **Step 1: New `compute` (replace the body after `assert real is not None`)**

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "median_dcr": 0.0, "p05_dcr": 0.0},
                notes=["no encodable columns"],
            )
        dists = nb.syn_to_real(1)
        median = float(np.median(dists))
        p05 = float(np.quantile(dists, 0.05))
        score = float(min(1.0, p05))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "median_dcr": median, "p05_dcr": p05},
            notes=["capped for gower" ] if nb.capped else [],
        )
```
Imports: replace `from sklearn.neighbors import NearestNeighbors` and `from syneva.compliance._encode import encode_pair` with `from syneva.core.neighbors import Neighbors`. Keep `import numpy as np`.

- [ ] **Step 2: Append gower smoke test** to `tests/unit/compliance/test_dcr.py`:

```python
def test_dcr_gower_backend_runs():
    import pandas as pd
    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": list(range(100)), "c": ["a", "b"] * 50})
    syn = pd.DataFrame({"x": list(range(1000, 1100)), "c": ["a", "b"] * 50})
    r = DCR(distance="gower").compute(real, syn, Metadata.infer(real))
    assert 0.0 <= r.scalars["score"] <= 1.0
    assert r.scalars["median_dcr"] >= 0.0
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/compliance/test_dcr.py --no-cov -v` (existing tests unchanged + new gower test all pass).

- [ ] **Step 4: Commit** — `git commit -m "refactor(compliance): dcr via Neighbors; gower option"`

### Task 4: `nndr`

**Files:** Modify `src/syneva/compliance/nndr.py`; Test append `tests/unit/compliance/test_nndr.py`.

- [ ] **Step 1:** add the `__init__` and replace the `encode_pair`+`NearestNeighbors` block. The current logic: `d_real = nn_real.kneighbors(X_syn)` (per syn, nearest real) and `d_syn = NN(2).fit(X_syn).kneighbors(X_syn)[:,1]` (within-syn). New body:

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["insufficient data"],
            )
        d_real = nb.syn_to_real(1)
        d_syn = nb.syn_self(1)
        ratio = d_real / np.where(d_syn > 0, d_syn, 1e-9)
        med = float(np.median(ratio))
        score = float(min(1.0, med))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nndr_median": med},
            notes=["capped for gower"] if nb.capped else [],
        )
```
(Match the exact current scalar keys/guards in the file; if the current guard differs, preserve its behaviour. Keep `import numpy as np`; swap imports to `Neighbors`.)

- [ ] **Step 2: Append gower smoke test** to `tests/unit/compliance/test_nndr.py`:

```python
def test_nndr_gower_backend_runs(real_df, syn_good_df, metadata):
    r = NNDR(distance="gower").compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["score"] <= 1.0
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/compliance/test_nndr.py --no-cov -v` (existing unchanged + new pass).
- [ ] **Step 4: Commit** — `git commit -m "refactor(compliance): nndr via Neighbors; gower option"`

### Task 5: `nn_adversarial_accuracy`

**Files:** Modify `src/syneva/coverage/extended/nn_adversarial_accuracy.py`; Test append `tests/unit/coverage/extended/test_nn_adversarial_accuracy.py`.

- [ ] **Step 1:** add `__init__`, replace block. Current: `d_rr=real_self`, `d_ss=syn_self`, `d_rs=real_to_syn`, `d_sr=syn_to_real`. New body:

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nn_adversarial_accuracy": 0.5},
                notes=["insufficient data"],
            )
        d_rr = nb.real_self(1)
        d_ss = nb.syn_self(1)
        d_rs = nb.real_to_syn(1)
        d_sr = nb.syn_to_real(1)
        aa = 0.5 * (float(np.mean(d_rs > d_rr)) + float(np.mean(d_sr > d_ss)))
        score = float(min(1.0, max(0.0, 1.0 - 2.0 * abs(aa - 0.5))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nn_adversarial_accuracy": float(aa)},
            notes=["capped for gower"] if nb.capped else [],
        )
```

- [ ] **Step 2: Append gower smoke test:**

```python
def test_nnaa_gower_backend_runs():
    import numpy as np
    import pandas as pd
    from syneva.core.metadata import Metadata

    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=80), "c": ["a", "b"] * 40})
    syn = pd.DataFrame({"x": rng.normal(size=80), "c": ["a", "b"] * 40})
    r = NNAdversarialAccuracy(distance="gower").compute(real, syn, Metadata.infer(real))
    assert 0.0 <= r.scalars["score"] <= 1.0
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/coverage/extended/test_nn_adversarial_accuracy.py --no-cov -v`.
- [ ] **Step 4: Commit** — `git commit -m "refactor(coverage): nn_adversarial_accuracy via Neighbors; gower option"`

### Task 6: `authenticity`

**Files:** Modify `src/syneva/coverage/extended/authenticity.py`; Test append `tests/unit/coverage/extended/test_authenticity.py`.

- [ ] **Step 1:** add `__init__`, replace block. Current: `d_syn_to_real = syn_to_real(1)`, `d_real_self = real_self(1)`, `eps = 0.05*median(d_real_self)`, `auth = mean(d_syn_to_real > eps)`. New body (preserve the exact eps/auth logic already in the file):

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "authenticity": 1.0},
                notes=["insufficient data"],
            )
        d_syn_to_real = nb.syn_to_real(1)
        d_real_self = nb.real_self(1)
        eps = 0.05 * float(np.median(d_real_self))
        auth = float(np.mean(d_syn_to_real > eps))
        return MetricResult(
            spec=self.spec,
            scalars={"score": auth, "authenticity": auth},
            notes=["capped for gower"] if nb.capped else [],
        )
```
(Confirm the eps factor matches the current file — it is `0.05 * median(real-self NN)`; keep it identical.)

- [ ] **Step 2: Append gower smoke test:**

```python
def test_authenticity_gower_backend_runs(real_df, syn_good_df, metadata):
    r = Authenticity(distance="gower").compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["authenticity"] <= 1.0
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/coverage/extended/test_authenticity.py --no-cov -v`.
- [ ] **Step 4: Commit** — `git commit -m "refactor(coverage): authenticity via Neighbors; gower option"`

### Task 7: `alpha_precision_beta_recall`

**Files:** Modify `src/syneva/coverage/extended/alpha_precision.py`; Test append `tests/unit/coverage/extended/test_alpha_precision.py`.

- [ ] **Step 1:** add `__init__`, replace block. Current logic: `k = max(1, min(5, len(X_real)-1))`; `radii_real = real_self(k)`; `radii_syn = syn_self(k)`; `d_to_real = syn_to_real(1)`; `alpha = quantile(radii_real, 0.9)`; `precision = mean(d_to_real <= alpha)`; `d_to_syn = real_to_syn(1)`; `beta = quantile(radii_syn, 0.9)`; `recall = mean(d_to_syn <= beta)`. New body:

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "alpha_precision": 1.0, "beta_recall": 1.0},
                notes=["no encodable columns"],
            )
        k = max(1, min(5, len(real) - 1))
        radii_real = nb.real_self(k)
        radii_syn = nb.syn_self(k)
        d_to_real = nb.syn_to_real(1)
        alpha = np.quantile(radii_real, 0.9)
        precision = float(np.mean(d_to_real <= alpha))
        d_to_syn = nb.real_to_syn(1)
        beta = np.quantile(radii_syn, 0.9)
        recall = float(np.mean(d_to_syn <= beta))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float((precision + recall) / 2),
                "alpha_precision": precision,
                "beta_recall": recall,
            },
            notes=["capped for gower"] if nb.capped else [],
        )
```
**Note on equivalence:** the current code uses `len(X_real)` for the `k` cap; here we use `len(real)`. For euclidean these are equal (encode_pair preserves row count), so the existing test is preserved. Verify the existing test still passes; if it asserts an exact value sensitive to `k`, confirm `len(real) == len(X_real)`.

- [ ] **Step 2: Append gower smoke test:**

```python
def test_alpha_precision_gower_backend_runs():
    import numpy as np
    import pandas as pd
    from syneva.core.metadata import Metadata

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=120), "c": ["a", "b", "c"] * 40})
    r = AlphaPrecisionBetaRecall(distance="gower").compute(df, df, Metadata.infer(df))
    assert r.scalars["alpha_precision"] > 0.5
    assert r.scalars["beta_recall"] > 0.5
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/coverage/extended/test_alpha_precision.py --no-cov -v`.
- [ ] **Step 4: Commit** — `git commit -m "refactor(coverage): alpha-precision/beta-recall via Neighbors; gower option"`

### Task 8: `epsilon_identifiability`

**Files:** Modify `src/syneva/compliance/extended/epsilon_identifiability.py`; Test append `tests/unit/compliance/extended/test_epsilon_identifiability.py`.

- [ ] **Step 1:** add `__init__`, replace block. Current: `d_self = real_self(1)`, `d_syn = real_to_syn(1)`, `risk = mean(d_syn <= d_self)`. New body:

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance)
        if nb.n_features == 0 or len(real) < 2 or len(synthetic) < 1:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "identifiability_risk": 0.0},
                notes=["insufficient data"],
            )
        d_self = nb.real_self(1)
        d_syn = nb.real_to_syn(1)
        risk = float(np.mean(d_syn <= d_self))
        score = float(min(1.0, max(0.0, 1.0 - risk)))
        notes = ["unweighted distance; entropy-weighting is a planned refinement"]
        if nb.capped:
            notes.append("capped for gower")
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "identifiability_risk": risk},
            notes=notes,
        )
```

- [ ] **Step 2: Append gower smoke test:**

```python
def test_epsilon_gower_backend_runs():
    import numpy as np
    import pandas as pd
    from syneva.core.metadata import Metadata

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "c": ["a", "b"] * 50})
    r = EpsilonIdentifiability(distance="gower").compute(df, df, Metadata.infer(df))
    assert r.scalars["identifiability_risk"] > 0.5  # syn == real -> identifiable
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/compliance/extended/test_epsilon_identifiability.py --no-cov -v` (incl. the existing duplicate-row test).
- [ ] **Step 4: Commit** — `git commit -m "refactor(compliance): epsilon_identifiability via Neighbors; gower option"`

### Task 9: `attribute_disclosure`

**Files:** Modify `src/syneva/compliance/extended/attribute_disclosure.py`; Test append `tests/unit/compliance/extended/test_attribute_disclosure.py`.

- [ ] **Step 1:** add `__init__`, replace the `encode_pair(... qi_meta)` + `NearestNeighbors` index block with a `Neighbors` built on `qi_meta`. Keep the sensitive/QI selection, the `qi_meta` construction, the no-sensitive skip, and the per-sensitive-column match logic UNCHANGED. New neighbour part:

```python
    def __init__(self, distance: str = "euclidean") -> None:
        self.distance = distance

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        sensitive = [n for n, cm in meta.columns.items() if cm.sensitive]
        qi = [n for n, cm in meta.columns.items() if not cm.sensitive and cm.dtype in _USABLE]
        if not sensitive or not qi:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no sensitive columns or no quasi-identifiers; skipped"],
            )
        qi_meta = Metadata(columns={n: meta.columns[n] for n in qi})
        nb = Neighbors(real, synthetic, qi_meta, distance=self.distance)
        if nb.n_features == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no encodable quasi-identifiers; skipped"],
            )
        idx = nb.real_to_syn_index()
        real_r = real.reset_index(drop=True)
        syn_r = synthetic.reset_index(drop=True)
        matches = np.ones(len(real_r), dtype=bool)
        for n in sensitive:
            guessed = syn_r[n].to_numpy()[idx]
            actual = real_r[n].to_numpy()
            if meta.columns[n].dtype is ColumnType.NUMERIC:
                col = pd.to_numeric(real_r[n], errors="coerce")
                thr = float(col.max() - col.min()) / 30.0
                g = pd.to_numeric(pd.Series(guessed), errors="coerce").to_numpy()
                a = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy()
                col_match = (np.abs(g - a) <= thr) if thr > 0 else (g == a)
            else:
                g = pd.Series(guessed).astype("string").fillna("__NA__").to_numpy()
                a = pd.Series(actual).astype("string").fillna("__NA__").to_numpy()
                col_match = g == a
            matches &= col_match
        disclosure_rate = float(np.mean(matches))
        score = float(min(1.0, max(0.0, 1.0 - disclosure_rate)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "disclosure_rate": disclosure_rate},
            notes=["capped for gower"] if nb.capped else [],
        )
```
Imports: drop `encode_pair`/`NearestNeighbors`, add `from syneva.core.neighbors import Neighbors`; keep `numpy`, `pandas`, `ColumnType`, `Metadata`, `_USABLE`.

**Gower caveat:** when gower caps the synthetic frame, `idx` references the capped synthetic rows, but the sensitive lookup uses the un-capped `syn_r`. To keep correctness simple, in the gower path the cap must be large enough or disabled for this metric. Set `Neighbors(..., cap=10**9)` here (no cap for attribute_disclosure) so `idx` indexes the full `synthetic`. (Document this: attribute_disclosure does not cap; it is O(n_real × n_syn) in gower.)

- [ ] **Step 2: Append gower smoke test:**

```python
def test_attribute_disclosure_gower_backend_runs():
    import pandas as pd
    from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

    qi = list(range(60))
    secret = ["A" if v < 30 else "B" for v in qi]
    df = pd.DataFrame({"qi": qi, "secret": secret})
    meta = Metadata(
        columns={
            "qi": ColumnMetadata(name="qi", dtype=ColumnType.NUMERIC),
            "secret": ColumnMetadata(name="secret", dtype=ColumnType.CATEGORICAL, sensitive=True),
        }
    )
    r = AttributeDisclosure(distance="gower").compute(df, df, meta)
    assert r.scalars["disclosure_rate"] > 0.9
```

- [ ] **Step 3: Run** — `uv run pytest tests/unit/compliance/extended/test_attribute_disclosure.py --no-cov -v`.
- [ ] **Step 4: Commit** — `git commit -m "refactor(compliance): attribute_disclosure via Neighbors; gower option"`

---

## Task 10: Runner threading + integration + full suite

**Files:**
- Modify: `src/syneva/core/runner.py`
- Test: `tests/integration/test_gower_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_gower_backend.py
import pytest

import syneva
from syneva.core.errors import SynevaError


def test_gower_distance_runs_end_to_end(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core", "extended"), distance="gower")
    by = {r.spec.name: r for r in rep.results}
    for m in ["dcr", "nndr", "nn_adversarial_accuracy", "authenticity",
              "alpha_precision_beta_recall", "epsilon_identifiability"]:
        assert m in by
        assert by[m].error is None, f"{m} errored under gower: {by[m].error}"


def test_invalid_distance_raises(real_df, syn_good_df, metadata):
    with pytest.raises(SynevaError, match="distance"):
        syneva.evaluate(real_df, syn_good_df, metadata, distance="manhattan")
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/integration/test_gower_backend.py --no-cov -v` (TypeError: unexpected `distance`).

- [ ] **Step 3: Thread `distance` through the runner** in `src/syneva/core/runner.py`:

(a) Generalize `_instantiate` — add a `distance` parameter and pass it when accepted:

```python
def _instantiate(cls, utility_tasks, fairness_specs, distance, random_state):
    """Instantiate a metric, passing only the kwargs its constructor accepts."""
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "tasks" in params:
        kwargs["tasks"] = utility_tasks
    if "specs" in params:
        kwargs["specs"] = fairness_specs if fairness_specs is not None else []
    if "distance" in params:
        kwargs["distance"] = distance
    if "random_state" in params:
        kwargs["random_state"] = random_state
    return cls(**kwargs)
```

(b) Add `distance: str = "euclidean"` to BOTH `evaluate(...)` and `evaluate_with(...)` signatures (after `run_fairness`), validate it in `evaluate_with` near the top (after the imports line):

```python
    if distance not in ("euclidean", "gower"):
        raise SynevaError(f"unknown distance '{distance}'; use 'euclidean' or 'gower'")
```

(c) Forward `distance=distance` in the `evaluate -> evaluate_with` call.

(d) Update the instantiation call site to `inst = _instantiate(cls, utility_tasks, fairness_specs, distance, random_state)`.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/integration/test_gower_backend.py --no-cov -v` (2 passed).

- [ ] **Step 5: Full suite + behaviour-preservation check**

Run: `uv run pytest -q`
Expected: ALL pass (the existing euclidean metric tests and the golden snapshot unchanged), coverage >= 85%. If the golden test fails, the euclidean refactor changed values — STOP and report; do NOT `--update-golden`.

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(runner): distance=euclidean|gower option threaded to NN metrics"
```

---

## Self-review notes

- **Spec coverage:** `gower_matrix` (T1) · `Neighbors` euclidean+gower (T2) · all 7 NN metrics refactored with `distance` field (T3–T9) · runner `distance` arg + `_instantiate` + validation (T10) · euclidean-equivalence test (T2) · per-metric gower smoke (T3–T9) · integration + invalid-distance (T10) · behaviour-preservation guardrail = existing tests + golden unchanged (every refactor task + T10 Step 5).
- **Type/signature consistency:** `Neighbors(real, synthetic, meta, *, distance, cap, random_state)` and its methods (`real_self`/`syn_self`/`real_to_syn`/`syn_to_real`/`real_to_syn_index`/`n_features`) are used identically across T3–T9; every metric gains `__init__(self, distance="euclidean")`; `_instantiate(cls, utility_tasks, fairness_specs, distance, random_state)` matches its call site.
- **No placeholders:** every step has complete code. The one explicit deviation is `attribute_disclosure` using `cap=10**9` (no cap) in gower so the nearest-syn index aligns with the full synthetic frame — documented in Task 9.
