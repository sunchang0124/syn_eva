# Holdout Test-Set Support (Phase 0c-b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `holdout` real test set to `evaluate()` that utility metrics test on, MIA uses as non-members, and DCR/NNDR use as a privacy baseline — fully behavior-preserving when `holdout=None`.

**Architecture:** A new `encode_frames` gives a consistent 3-frame encoding; `Neighbors` gains an optional `holdout` (→ `holdout_to_real`/`holdout_self`); five metrics (`tstr_suite`, `multi_target_utility`, `mia_auc`, `dcr`, `nndr`) gain a `holdout` constructor field with a conditional branch; the runner threads `holdout` via `_instantiate`.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-23-syneva-parity-0c-holdout.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Behaviour preservation:** every change is conditional on `holdout`/`self.holdout` being non-None. With no holdout, all 5 metrics behave as today — their existing tests must pass UNCHANGED and the golden snapshot must stay green (NEVER `--update-golden`). If an existing test changes, stop and report.
- `_EPS = 1e-12`; clamp with `float(min(1.0, max(0.0, x)))`.

## File structure
- Modify: `src/syneva/compliance/_encode.py` (add `encode_frames`)
- Modify: `src/syneva/core/neighbors.py` (holdout support)
- Modify: `src/syneva/utility/tstr.py`, `src/syneva/utility/extended/multi_target.py`
- Modify: `src/syneva/compliance/extended/mia.py`, `src/syneva/compliance/dcr.py`, `src/syneva/compliance/nndr.py`
- Modify: `src/syneva/core/runner.py`, `src/syneva/ui/core.py`, `src/syneva/ui/app.py`
- Tests under `tests/unit/compliance/`, `tests/unit/core/`, `tests/unit/utility/`, `tests/integration/`.

---

## Task 1: `encode_frames` (consistent N-frame encoding)

**Files:** Modify `src/syneva/compliance/_encode.py`; Test `tests/unit/compliance/test_encode_frames.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/compliance/test_encode_frames.py
import numpy as np
import pandas as pd

from syneva.compliance._encode import encode_frames, encode_pair
from syneva.core.metadata import Metadata


def test_two_frames_match_encode_pair():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    syn = pd.DataFrame({"x": [2.0, 3.0], "c": ["b", "a"]})
    meta = Metadata.infer(real)
    xr, xs = encode_pair(real, syn, meta)
    fr, fs = encode_frames([real, syn], meta)
    assert np.allclose(fr, xr)
    assert np.allclose(fs, xs)


def test_three_frames_share_space_identical_row_same_vector():
    real = pd.DataFrame({"x": [0.0, 10.0], "c": ["a", "b"]})
    syn = pd.DataFrame({"x": [5.0], "c": ["a"]})
    hold = pd.DataFrame({"x": [0.0], "c": ["a"]})  # identical to real row 0
    meta = Metadata.infer(real)
    xr, xs, xh = encode_frames([real, syn, hold], meta)
    assert xr.shape[1] == xs.shape[1] == xh.shape[1]
    assert np.allclose(xr[0], xh[0])  # same raw row -> same encoded vector
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/test_encode_frames.py --no-cov -v`.

- [ ] **Step 3: Implement** — append to `src/syneva/compliance/_encode.py` (it already imports `numpy as np`, `pandas as pd`, `StandardScaler`, `ColumnType`, `Metadata`):

```python
def encode_frames(frames: list[pd.DataFrame], meta: Metadata) -> list[np.ndarray]:
    """Encode several frames into ONE shared numeric space (the N-frame
    generalization of encode_pair): numeric columns standardized with a single
    scaler fit on the concatenation, categoricals one-hot over the union of
    categories. Returns one matrix per input frame, in order."""
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
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/compliance/test_encode_frames.py --no-cov -v` (2 passed).

- [ ] **Step 5: Commit** — `git commit -m "feat(compliance): encode_frames N-frame shared encoding"`

---

## Task 2: `Neighbors` holdout support

**Files:** Modify `src/syneva/core/neighbors.py`; Test append `tests/unit/core/test_neighbors.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/core/test_neighbors.py
def test_holdout_to_real_euclidean():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    syn = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    hold = pd.DataFrame({"x": rng.normal(size=20), "y": rng.normal(size=20)})
    meta = Metadata.infer(real)
    nb = Neighbors(real, syn, meta, distance="euclidean", holdout=hold)
    d = nb.holdout_to_real(1)
    assert d.shape == (20,)
    assert (d >= 0).all()
    ds = nb.holdout_self(1)
    assert ds.shape == (20,)


def test_holdout_methods_raise_without_holdout():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(size=10)})
    syn = pd.DataFrame({"x": rng.normal(size=10)})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="euclidean")
    with pytest.raises(ValueError, match="holdout"):
        nb.holdout_to_real(1)


def test_holdout_gower_runs():
    real = pd.DataFrame({"n": [0.0, 1.0, 2.0, 3.0], "c": ["a", "b", "a", "b"]})
    syn = pd.DataFrame({"n": [0.5, 1.5], "c": ["a", "b"]})
    hold = pd.DataFrame({"n": [0.2, 2.2], "c": ["a", "a"]})
    nb = Neighbors(real, syn, Metadata.infer(real), distance="gower", holdout=hold)
    assert nb.holdout_to_real(1).shape == (2,)
    assert nb.holdout_self(1).shape == (2,)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/core/test_neighbors.py -k holdout --no-cov -v`.

- [ ] **Step 3: Implement** — edit `src/syneva/core/neighbors.py`:

(a) Import `encode_frames` alongside the existing `encode_pair` import:
```python
from syneva.compliance._encode import encode_frames, encode_pair
```

(b) Add `holdout: pd.DataFrame | None = None` to `__init__` (after `synthetic`/`meta`, as a keyword-only param alongside `distance`). Replace the euclidean and gower branches of `__init__` to handle holdout:

```python
        if distance == "gower":
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
            self._d_rr = gower_matrix(r, r, meta)
            self._d_rs = gower_matrix(r, s, meta)
            self._d_ss = gower_matrix(s, s, meta)
            self._d_hr = gower_matrix(h, r, meta) if h is not None else None
            self._d_hh = gower_matrix(h, h, meta) if h is not None else None
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
```

(Also initialise `self._d_hr = None`, `self._d_hh = None`, `self._x_holdout = None` defaults are set in their branches above; ensure both branches assign them.)

(c) Add the two methods (after `real_to_syn_index`):

```python
    def holdout_to_real(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        if self.distance == "gower":
            d = self._d_hr
            if k > d.shape[1]:
                raise ValueError(f"k={k} too large: only {d.shape[1]} real rows")
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        return (
            NearestNeighbors(n_neighbors=k)
            .fit(self._x_real)
            .kneighbors(self._x_holdout)[0][:, k - 1]
        )

    def holdout_self(self, k: int = 1) -> np.ndarray:
        if not self._has_holdout:
            raise ValueError("no holdout provided to Neighbors")
        if self.distance == "gower":
            d = self._d_hh.copy()
            n = d.shape[0]
            if k >= n:
                raise ValueError(f"k={k} too large: only {n - 1} other holdout rows")
            np.fill_diagonal(d, np.inf)
            return np.partition(d, k - 1, axis=1)[:, k - 1]
        x = self._x_holdout
        return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[0][:, k]
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/core/test_neighbors.py --no-cov -v` (existing + new pass; the no-holdout euclidean equivalence test still passes, proving the 2-frame path is unchanged).

- [ ] **Step 5: Commit** — `git commit -m "feat(core): Neighbors holdout support (holdout_to_real/holdout_self)"`

---

## Task 3: `tstr_suite` holdout (+ `multi_target_utility` forwarding)

**Files:** Modify `src/syneva/utility/tstr.py`, `src/syneva/utility/extended/multi_target.py`; Test append `tests/unit/utility/test_tstr.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/utility/test_tstr.py
def test_tstr_uses_holdout_as_test_set(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=7)
    suite = TSTRSuite(
        tasks=[UtilityTask(target="high_income", task_type="classification")],
        holdout=holdout,
    )
    r = suite.compute(real_df, syn_good_df, metadata)
    assert "utility_ratio_high_income" in r.scalars
    assert r.scalars["utility_ratio_high_income"] >= 0.0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/utility/test_tstr.py -k holdout --no-cov -v` (TypeError: unexpected `holdout`).

- [ ] **Step 3: Implement** — in `src/syneva/utility/tstr.py`, add a field and a conditional test-set:

(a) Add to the dataclass fields (after `random_state`):
```python
    holdout: object | None = None  # pandas DataFrame test set, or None
```

(b) Inside the `for t in self.tasks:` loop, replace the `train_test_split(...)` + the two `_score(...)` calls with a holdout-aware version:
```python
            tt = t.task_type or "regression"
            X_syn = synthetic[features]
            y_syn = synthetic[t.target]
            if self.holdout is not None:
                X_test = self.holdout[features]
                y_test = self.holdout[t.target]
                X_real_tr, y_real_tr = real[features], real[t.target]
            else:
                X_real_tr, X_test, y_real_tr, y_test = train_test_split(
                    real[features],
                    real[t.target],
                    test_size=0.3,
                    random_state=self.random_state,
                    stratify=real[t.target] if t.task_type == "classification" else None,
                )
            trtr_score = _score(
                tt, build_pipeline(tt, meta, features, self.random_state),
                X_real_tr, y_real_tr, X_test, y_test,
            )
            tstr_score = _score(
                tt, build_pipeline(tt, meta, features, self.random_state),
                X_syn, y_syn, X_test, y_test,
            )
```
(Keep the rest — the ratio computation, scalars, per_task — unchanged. Remove the now-unused `X_real`/`y_real` locals.)

(c) In `src/syneva/utility/extended/multi_target.py`, add a `holdout` field and forward it. The class builds `TSTRSuite(tasks=tasks, random_state=self.random_state)` — change to `TSTRSuite(tasks=tasks, random_state=self.random_state, holdout=self.holdout)` and add `holdout: object | None = None` to its dataclass fields.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/utility/ --no-cov -v` (existing tests unchanged + new holdout test pass).

- [ ] **Step 5: Commit** — `git commit -m "feat(utility): TSTR/TRTR test on holdout when provided"`

---

## Task 4: `mia_auc` holdout non-members

**Files:** Modify `src/syneva/compliance/extended/mia.py`; Test append `tests/unit/compliance/extended/test_mia.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/compliance/extended/test_mia.py
def test_mia_with_holdout_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=3)
    r = MembershipInferenceAttack(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["mia_auc"] <= 1.0
    assert 0.0 <= r.scalars["score"] <= 1.0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/extended/test_mia.py -k holdout --no-cov -v` (TypeError: unexpected `holdout`).

- [ ] **Step 3: Implement** — edit `src/syneva/compliance/extended/mia.py`:

(a) Imports: add `from syneva.compliance._encode import encode_frames` (keep `encode_pair`).

(b) Add `__init__`:
```python
    def __init__(self, holdout: object | None = None) -> None:
        self.holdout = holdout
```

(c) In `compute`, branch on `self.holdout`:
```python
    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        if self.holdout is not None:
            x_real, x_syn, x_hold = encode_frames([real, synthetic, self.holdout], meta)
            if x_real.shape[1] == 0 or len(x_syn) < 1 or len(x_hold) < 1 or len(x_real) < 1:
                return MetricResult(
                    spec=self.spec, scalars={"score": 1.0, "mia_auc": 0.5}, notes=["insufficient data"]
                )
            nn = NearestNeighbors(n_neighbors=1).fit(x_syn)
            d_mem, _ = nn.kneighbors(x_real)
            d_non, _ = nn.kneighbors(x_hold)
            y = np.concatenate([np.ones(len(x_real)), np.zeros(len(x_hold))])
            scores = -np.concatenate([d_mem.ravel(), d_non.ravel()])
        else:
            x_real, x_syn = encode_pair(real, synthetic, meta)
            if x_real.shape[1] == 0 or len(x_real) < 4:
                return MetricResult(
                    spec=self.spec, scalars={"score": 1.0, "mia_auc": 0.5}, notes=["insufficient data"]
                )
            members, non_members = train_test_split(x_real, test_size=0.5, random_state=42)
            nn = NearestNeighbors(n_neighbors=1).fit(x_syn)
            d_mem, _ = nn.kneighbors(members)
            d_non, _ = nn.kneighbors(non_members)
            y = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
            scores = -np.concatenate([d_mem.ravel(), d_non.ravel()])
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.5
        score = float(1.0 - max(0.0, auc - 0.5) * 2)
        return MetricResult(spec=self.spec, scalars={"score": score, "mia_auc": auc})
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/compliance/extended/test_mia.py --no-cov -v`.

- [ ] **Step 5: Commit** — `git commit -m "feat(compliance): MIA uses holdout as non-members when provided"`

---

## Task 5: `dcr` holdout baseline

**Files:** Modify `src/syneva/compliance/dcr.py`; Test append `tests/unit/compliance/test_dcr.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/compliance/test_dcr.py
def test_dcr_holdout_baseline_flags_memorization():
    import pandas as pd
    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": [float(v) for v in range(100)]})
    syn = real.copy()  # verbatim copy => far closer to real than a disjoint holdout
    hold = pd.DataFrame({"x": [float(v) for v in range(1000, 1100)]})
    r = DCR(holdout=hold).compute(real, syn, Metadata.infer(real))
    assert "p05_dcr_holdout" in r.scalars
    assert r.scalars["score"] < 0.5  # synthetic far closer than holdout -> risk


def test_dcr_holdout_safe_when_synthetic_like_holdout():
    import pandas as pd
    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": [float(v) for v in range(100)]})
    syn = pd.DataFrame({"x": [v + 500.0 for v in range(100)]})
    hold = pd.DataFrame({"x": [v + 500.0 for v in range(100)]})  # syn as far as holdout
    r = DCR(holdout=hold).compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] > 0.8
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/test_dcr.py -k holdout --no-cov -v`.

- [ ] **Step 3: Implement** — edit `src/syneva/compliance/dcr.py`. Add `holdout` to `__init__`, build `Neighbors` with it, branch the score:

```python
    def __init__(self, distance: str = "euclidean", holdout: object | None = None) -> None:
        self.distance = distance
        self.holdout = holdout

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance, holdout=self.holdout)
        if nb.n_features == 0 or len(real) == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "median_dcr": 0.0, "p05_dcr": 0.0},
                notes=["no encodable columns"],
            )
        dists = nb.syn_to_real(1)
        median = float(np.median(dists))
        p05 = float(np.quantile(dists, 0.05))
        notes = ["capped for gower"] if nb.capped else []
        if self.holdout is not None and len(self.holdout) > 0:
            d_hold = nb.holdout_to_real(1)
            p05_hold = float(np.quantile(d_hold, 0.05))
            median_hold = float(np.median(d_hold))
            score = float(min(1.0, max(0.0, p05 / (p05_hold + 1e-12))))
            return MetricResult(
                spec=self.spec,
                scalars={
                    "score": score,
                    "median_dcr": median,
                    "p05_dcr": p05,
                    "median_dcr_holdout": median_hold,
                    "p05_dcr_holdout": p05_hold,
                },
                notes=notes,
            )
        score = float(min(1.0, p05))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "median_dcr": median, "p05_dcr": p05},
            notes=notes,
        )
```

Add metric-info `_SCALAR_LABELS` entries in `src/syneva/core/metric_info.py`: `"median_dcr_holdout": "Median distance (holdout)",` and `"p05_dcr_holdout": "5th-percentile distance (holdout)",`.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/compliance/test_dcr.py tests/unit/core/test_metric_info.py --no-cov -v` (existing DCR tests unchanged + new holdout tests pass).

- [ ] **Step 5: Commit** — `git commit -m "feat(compliance): dcr holdout baseline (relative privacy score)"`

---

## Task 6: `nndr` holdout baseline

**Files:** Modify `src/syneva/compliance/nndr.py`; Test append `tests/unit/compliance/test_nndr.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/compliance/test_nndr.py
def test_nndr_holdout_baseline_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=5)
    r = NNDR(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert "nndr_median_holdout" in r.scalars
    assert 0.0 <= r.scalars["score"] <= 1.0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/test_nndr.py -k holdout --no-cov -v`.

- [ ] **Step 3: Implement** — edit `src/syneva/compliance/nndr.py`:

```python
    def __init__(self, distance: str = "euclidean", holdout: object | None = None) -> None:
        self.distance = distance
        self.holdout = holdout

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        nb = Neighbors(real, synthetic, meta, distance=self.distance, holdout=self.holdout)
        if nb.n_features == 0 or len(real) == 0 or len(synthetic) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nndr_median": 1.0},
                notes=["insufficient data"],
            )
        ratio_syn = nb.syn_to_real(1) / np.where(nb.syn_self(1) > 0, nb.syn_self(1), 1e-9)
        med = float(np.median(ratio_syn))
        notes = ["capped for gower"] if nb.capped else []
        if self.holdout is not None and len(self.holdout) > 1:
            hs = nb.holdout_self(1)
            ratio_hold = nb.holdout_to_real(1) / np.where(hs > 0, hs, 1e-9)
            med_hold = float(np.median(ratio_hold))
            score = float(min(1.0, max(0.0, med / (med_hold + 1e-12))))
            return MetricResult(
                spec=self.spec,
                scalars={"score": score, "nndr_median": med, "nndr_median_holdout": med_hold},
                notes=notes,
            )
        score = float(min(1.0, med))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nndr_median": med},
            notes=notes,
        )
```

Add metric-info `_SCALAR_LABELS` entry: `"nndr_median_holdout": "Median distance ratio (holdout)",`.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/compliance/test_nndr.py tests/unit/core/test_metric_info.py --no-cov -v`.

- [ ] **Step 5: Commit** — `git commit -m "feat(compliance): nndr holdout baseline (relative privacy score)"`

---

## Task 7: Runner `holdout` threading + integration

**Files:** Modify `src/syneva/core/runner.py`; Test `tests/integration/test_holdout.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_holdout.py
import pytest

import syneva
from syneva import UtilityTask
from syneva.core.errors import SynevaError


def test_holdout_runs_end_to_end(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=2)
    rep = syneva.evaluate(
        real_df, syn_good_df, metadata,
        tiers=("core", "extended"),
        holdout=holdout,
        run_utility=True,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert by["dcr"].error is None and "p05_dcr_holdout" in by["dcr"].scalars
    assert by["mia_auc"].error is None
    assert by["tstr_suite"].error is None


def test_holdout_schema_mismatch_raises(real_df, syn_good_df, metadata):
    bad = real_df.drop(columns=[real_df.columns[0]])
    with pytest.raises(SynevaError, match="holdout"):
        syneva.evaluate(real_df, syn_good_df, metadata, holdout=bad)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/integration/test_holdout.py --no-cov -v` (unexpected `holdout`).

- [ ] **Step 3: Implement** — edit `src/syneva/core/runner.py`:

(a) `_instantiate` gains `holdout`:
```python
def _instantiate(cls, utility_tasks, fairness_specs, distance, holdout, random_state):
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "tasks" in params:
        kwargs["tasks"] = utility_tasks
    if "specs" in params:
        kwargs["specs"] = fairness_specs if fairness_specs is not None else []
    if "distance" in params:
        kwargs["distance"] = distance
    if "holdout" in params:
        kwargs["holdout"] = holdout
    if "random_state" in params:
        kwargs["random_state"] = random_state
    return cls(**kwargs)
```

(b) Add `holdout: pd.DataFrame | None = None` to BOTH `evaluate` and `evaluate_with` (after `distance`); forward `holdout=holdout` in the `evaluate -> evaluate_with` call.

(c) In `evaluate_with`, after the distance validation, validate holdout columns:
```python
    if holdout is not None and set(holdout.columns) != set(synthetic.columns):
        raise SynevaError("holdout columns must match the synthetic/real columns")
```

(d) Update the call site: `inst = _instantiate(cls, utility_tasks, fairness_specs, distance, holdout, random_state)`.

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/integration/test_holdout.py --no-cov -v` (2 passed).

- [ ] **Step 5: Full suite** — `uv run pytest -q` → all pass, golden UNCHANGED, coverage >= 85%. If golden fails, STOP (a no-holdout path changed).

- [ ] **Step 6: Commit** — `git commit -m "feat(runner): holdout test-set threaded through evaluate()"`

---

## Task 8: UI holdout uploader

**Files:** Modify `src/syneva/ui/core.py`, `src/syneva/ui/app.py`; Test append `tests/unit/ui/test_core.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/ui/test_core.py
def test_run_report_accepts_holdout():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    hold = real.sample(frac=0.3, random_state=1)
    rep = core.run_report(real, syn, Metadata.infer(real), ["dcr"], holdout=hold)
    by = {r.spec.name: r for r in rep.results}
    assert "p05_dcr_holdout" in by["dcr"].scalars
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/ui/test_core.py -k holdout --no-cov -v`.

- [ ] **Step 3: Implement**

(a) `src/syneva/ui/core.py` — `run_report` gains `holdout: object | None = None` (after `fairness_specs`) and forwards `holdout=holdout` to `evaluate_with(...)`.

(b) `src/syneva/ui/app.py` `_sidebar()` — in the "1 · Upload data" section, add a third optional uploader after the synthetic one:
```python
        holdout_file = st.file_uploader(
            "Holdout / test data (optional, CSV/Parquet)", type=["csv", "parquet"]
        )
```
After the real/synthetic load `try` block, load the holdout if present:
```python
        holdout = None
        if holdout_file is not None:
            try:
                holdout = core.load_table(holdout_file)
            except ValueError as e:
                st.error(str(e))
                return None
```
Add `"holdout": holdout,` to the returned dict; in `main()` pass `holdout=cfg["holdout"]` to `core.run_report(...)`.

- [ ] **Step 4: Run + boot smoke** — `uv run pytest tests/unit/ui/test_core.py tests/unit/ui/test_app_importable.py --no-cov -v`; then `uv run python -c "import syneva.ui.app"`; headless boot on port 8604 (200, no tracebacks), then kill.

- [ ] **Step 5: Full suite** — `uv run pytest -q` → pass, golden unchanged, coverage >= 85%.

- [ ] **Step 6: Commit** — `git commit -m "feat(ui): optional holdout uploader"`

---

## Self-review notes
- **Spec coverage:** encode_frames (T1) · Neighbors holdout + holdout_to_real/holdout_self (T2) · utility holdout test set (T3) · MIA non-members (T4) · DCR baseline (T5) · NNDR baseline (T6) · runner threading + validation (T7) · UI (T8). Behaviour preservation guarded in every task (holdout=None unchanged; existing tests + golden green) and checked full-suite in T7/T8.
- **Type/signature consistency:** `Neighbors(..., holdout=None)` + `holdout_to_real`/`holdout_self`; `_instantiate(cls, utility_tasks, fairness_specs, distance, holdout, random_state)` matches its call site; every metric gains a `holdout` constructor field; new scalar keys (`*_holdout`) all get `_SCALAR_LABELS` entries (T5/T6).
- **No placeholders:** every step has complete code.
