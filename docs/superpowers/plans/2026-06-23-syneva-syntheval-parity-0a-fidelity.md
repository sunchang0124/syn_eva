# SynthEval Parity 0a — Fidelity Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add seven fidelity/coverage metrics so syneva's Congruence + Coverage dimensions fully cover SynthEval, each with a normalized 0–1 score, correct raw-direction hint, and full name + description, plus a CI-enforced parity manifest.

**Architecture:** Each metric is a self-contained `@registry.register` class following the existing pattern (a `ClassVar[MetricSpec]` + `compute(self, real, synthetic, meta) -> MetricResult`), auto-registered through the existing package `__init__` side-effect imports. Table-level metrics reuse `syneva.compliance._encode.encode_pair`. Every metric also gets entries in `src/syneva/core/metric_info.py` (full name, neutral description, raw-direction hint, scalar labels), which the existing guard tests enforce.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, scikit-learn (already dependencies), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-23-syneva-syntheval-parity-0a-fidelity.md`.

---

## CRITICAL environment notes (read before every task)

- Use `uv`: `uv run pytest ...`.
- The repo's pytest `addopts` includes `--cov-fail-under=85`, so running a SUBSET of tests reports a spurious coverage failure even when tests pass. ALWAYS add `--no-cov` for targeted runs (e.g. `uv run pytest tests/unit/congruence/test_dimension_wise_means.py --no-cov -v`). Only the full `uv run pytest` enforces coverage.
- After registering a new metric you MUST add its `metric_info.py` entries in the SAME task, or the full-registry guard tests (`tests/unit/core/test_metric_info.py`) fail. Each task re-runs that guard file to confirm.
- Before committing: run `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, then commit. If a commit aborts because a pre-commit hook modified files, `git add -A` and re-run the same commit.
- All scores are floats in `[0, 1]` with 1.0 = ideal. `_EPS = 1e-12` guards division. Clamp helper pattern used in the repo: `float(min(1.0, max(0.0, x)))`.

---

## File structure

- Create: `src/syneva/congruence/dimension_wise_means.py`, `ci_overlap.py`, `hellinger.py`, `quantile_mse.py`
- Create: `src/syneva/congruence/extended/mutual_information_difference.py`, `mmd.py`
- Create: `src/syneva/coverage/extended/nn_adversarial_accuracy.py`
- Modify: `src/syneva/congruence/__init__.py`, `src/syneva/congruence/extended/__init__.py`, `src/syneva/coverage/extended/__init__.py` (side-effect imports)
- Modify: `src/syneva/core/metric_info.py` (METRIC_NAMES, METRIC_INFO, RAW_HINT, _SCALAR_LABELS)
- Create: `tests/unit/congruence/test_dimension_wise_means.py`, `test_ci_overlap.py`, `test_hellinger.py`, `test_quantile_mse.py`
- Create: `tests/unit/congruence/extended/test_mutual_information_difference.py`, `test_mmd.py`
- Create: `tests/unit/coverage/extended/test_nn_adversarial_accuracy.py`
- Create: `tests/parity/__init__.py`, `tests/parity/syntheval_manifest.py`, `tests/parity/test_syntheval_parity.py`

---

## Task 1: `dimension_wise_means` (Congruence, core)

**Files:**
- Create: `src/syneva/congruence/dimension_wise_means.py`
- Modify: `src/syneva/congruence/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/test_dimension_wise_means.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/test_dimension_wise_means.py
import pandas as pd

from syneva.congruence.dimension_wise_means import DimensionWiseMeans
from syneva.core.metadata import Metadata


def test_identical_data_scores_high():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    r = DimensionWiseMeans().compute(df, df, Metadata.infer(df))
    assert r.scalars["score"] > 0.95
    assert r.scalars["mean_abs_std_diff"] < 0.05


def test_mean_shift_lowers_score():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    syn = pd.DataFrame({"x": [11.0, 12.0, 13.0, 14.0, 15.0]})
    r = DimensionWiseMeans().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_abs_std_diff"] > 1.0
    assert r.scalars["score"] < 0.5
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/test_dimension_wise_means.py --no-cov -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'syneva.congruence.dimension_wise_means'`.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/dimension_wise_means.py
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12


@registry.register
class DimensionWiseMeans:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="dimension_wise_means",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            d = abs(float(r.mean()) - float(s.mean())) / (float(r.std()) + _EPS)
            per_column[name] = {"std_mean_diff": d}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )
        mean_d = sum(v["std_mean_diff"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_d)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_abs_std_diff": float(mean_d)},
            per_column=per_column,
            notes=notes,
        )
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/__init__.py`, add (keep imports sorted; ruff will reorder):

```python
from syneva.congruence import (
    dimension_wise_means as dimension_wise_means,  # side-effect: registers DimensionWiseMeans
)
```

- [ ] **Step 5: Add metric-info entries**

In `src/syneva/core/metric_info.py` add, to the respective dicts:
- `METRIC_NAMES`: `"dimension_wise_means": "Dimension-wise means",`
- `METRIC_INFO`: `"dimension_wise_means": "Whether each numeric column's mean is preserved.",`
- `RAW_HINT`: `"dimension_wise_means": "Standardized gap between real and synthetic column means: 0 = identical, larger = more shifted. Lower is better.",`
- `_SCALAR_LABELS`: `"mean_abs_std_diff": "Mean standardized mean gap",`

- [ ] **Step 6: Run tests (metric + guards)**

Run: `uv run pytest tests/unit/congruence/test_dimension_wise_means.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass (the guard tests confirm the new metric has name/description/raw-hint and no direction claim in the description).

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence): dimension-wise means (SynthEval parity)"
```

---

## Task 2: `ci_overlap` (Congruence, core)

**Files:**
- Create: `src/syneva/congruence/ci_overlap.py`
- Modify: `src/syneva/congruence/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/test_ci_overlap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/test_ci_overlap.py
import numpy as np
import pandas as pd

from syneva.congruence.ci_overlap import CIOverlap
from syneva.core.metadata import Metadata


def test_identical_data_full_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300)})
    r = CIOverlap().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_ci_overlap"] > 0.95
    assert r.scalars["score"] > 0.95


def test_disjoint_means_no_overlap():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(loc=0, scale=1, size=300)})
    syn = pd.DataFrame({"x": rng.normal(loc=50, scale=1, size=300)})
    r = CIOverlap().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_ci_overlap"] < 0.05
    assert r.scalars["score"] < 0.05
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/test_ci_overlap.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/ci_overlap.py
from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12


@registry.register
class CIOverlap:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="ci_overlap",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if len(r) < 2 or len(s) < 2:
                notes.append(f"column '{name}' skipped (need >=2 values)")
                continue
            se_r = float(r.std()) / math.sqrt(len(r))
            se_s = float(s.std()) / math.sqrt(len(s))
            lo_r, hi_r = float(r.mean()) - 1.96 * se_r, float(r.mean()) + 1.96 * se_r
            lo_s, hi_s = float(s.mean()) - 1.96 * se_s, float(s.mean()) + 1.96 * se_s
            overlap = max(0.0, min(hi_r, hi_s) - max(lo_r, lo_s))
            avg_width = ((hi_r - lo_r) + (hi_s - lo_s)) / 2.0
            frac = float(min(1.0, max(0.0, overlap / (avg_width + _EPS))))
            per_column[name] = {"ci_overlap": frac}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )
        mean_overlap = sum(v["ci_overlap"] for v in per_column.values()) / len(per_column)
        return MetricResult(
            spec=self.spec,
            scalars={"score": float(mean_overlap), "mean_ci_overlap": float(mean_overlap)},
            per_column=per_column,
            notes=notes,
        )
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/__init__.py` add:

```python
from syneva.congruence import ci_overlap as ci_overlap  # side-effect: registers CIOverlap
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"ci_overlap": "Confidence-interval overlap",`
- `METRIC_INFO`: `"ci_overlap": "How much the 95% confidence intervals of column means overlap between real and synthetic data.",`
- `RAW_HINT`: `"ci_overlap": "Overlap fraction from 0 to 1: 1 = fully overlapping intervals, 0 = disjoint. Higher is better.",`
- `_SCALAR_LABELS`: `"mean_ci_overlap": "Mean CI overlap",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/congruence/test_ci_overlap.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence): confidence-interval overlap (SynthEval parity)"
```

---

## Task 3: `hellinger` (Congruence, core)

**Files:**
- Create: `src/syneva/congruence/hellinger.py`
- Modify: `src/syneva/congruence/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/test_hellinger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/test_hellinger.py
import pandas as pd

from syneva.congruence.hellinger import Hellinger
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta(cols):
    return Metadata(columns={n: ColumnMetadata(name=n, dtype=t) for n, t in cols.items()})


def test_identical_zero_distance():
    df = pd.DataFrame({"y": ["a", "b", "a", "b"] * 25, "x": list(range(100))})
    meta = _meta({"y": ColumnType.CATEGORICAL, "x": ColumnType.NUMERIC})
    r = Hellinger().compute(df, df, meta)
    assert r.scalars["mean_hellinger"] < 1e-9
    assert r.scalars["score"] > 0.99


def test_disjoint_categories_high_distance():
    real = pd.DataFrame({"y": ["a"] * 100})
    syn = pd.DataFrame({"y": ["b"] * 100})
    meta = _meta({"y": ColumnType.CATEGORICAL})
    r = Hellinger().compute(real, syn, meta)
    assert r.per_column["y"]["hellinger"] > 0.9
    assert r.scalars["score"] < 0.1
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/test_hellinger.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/hellinger.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_N_BINS = 20


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


@registry.register
class Hellinger:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="hellinger",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is ColumnType.CATEGORICAL:
                r = real[name].astype("string").fillna("__NA__").value_counts(normalize=True)
                s = synthetic[name].astype("string").fillna("__NA__").value_counts(normalize=True)
                cats = r.index.union(s.index)
                p = np.array([r.get(c, 0.0) for c in cats])
                q = np.array([s.get(c, 0.0) for c in cats])
                per_column[name] = {"hellinger": _hellinger(p, q)}
            elif cm.dtype is ColumnType.NUMERIC:
                r = pd.to_numeric(real[name], errors="coerce").dropna().to_numpy()
                s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
                if len(r) == 0 or len(s) == 0:
                    notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                    continue
                lo = float(min(r.min(), s.min()))
                hi = float(max(r.max(), s.max()))
                if hi <= lo:
                    per_column[name] = {"hellinger": 0.0}
                    continue
                bins = np.linspace(lo, hi, _N_BINS + 1)
                pr, _ = np.histogram(r, bins=bins)
                ps, _ = np.histogram(s, bins=bins)
                p = pr / pr.sum()
                q = ps / ps.sum()
                per_column[name] = {"hellinger": _hellinger(p, q)}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric or categorical columns"],
            )
        mean_h = sum(v["hellinger"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - mean_h)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_hellinger": float(mean_h)},
            per_column=per_column,
            notes=notes,
        )
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/__init__.py` add:

```python
from syneva.congruence import hellinger as hellinger  # side-effect: registers Hellinger
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"hellinger": "Hellinger distance",`
- `METRIC_INFO`: `"hellinger": "Per-column distributional distance between real and synthetic data.",`
- `RAW_HINT`: `"hellinger": "Ranges 0 to 1: 0 = identical distributions, larger = more different. Lower is better.",`
- `_SCALAR_LABELS`: `"mean_hellinger": "Mean Hellinger distance",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/congruence/test_hellinger.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence): Hellinger distance (SynthEval parity)"
```

---

## Task 4: `quantile_mse` (Congruence, core)

**Files:**
- Create: `src/syneva/congruence/quantile_mse.py`
- Modify: `src/syneva/congruence/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/test_quantile_mse.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/test_quantile_mse.py
import numpy as np
import pandas as pd

from syneva.congruence.quantile_mse import QuantileMSE
from syneva.core.metadata import Metadata


def test_identical_data_low_qmse():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    r = QuantileMSE().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_quantile_mse"] < 1e-6
    assert r.scalars["score"] > 0.99


def test_tail_shift_raises_qmse():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(loc=0, scale=1, size=500)})
    syn = pd.DataFrame({"x": rng.normal(loc=0, scale=4, size=500)})  # heavier tails
    r = QuantileMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_quantile_mse"] > 0.1
    assert r.scalars["score"] < 0.9
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/test_quantile_mse.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/quantile_mse.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12
_QUANTILES = np.arange(0.05, 1.0, 0.05)


@registry.register
class QuantileMSE:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="quantile_mse",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        for name, cm in meta.columns.items():
            if cm.dtype is not ColumnType.NUMERIC:
                continue
            r = pd.to_numeric(real[name], errors="coerce").dropna().to_numpy()
            s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
            if len(r) == 0 or len(s) == 0:
                notes.append(f"column '{name}' skipped (empty after NaN-drop)")
                continue
            scale = float(np.std(r)) + _EPS
            qr = np.quantile(r, _QUANTILES)
            qs = np.quantile(s, _QUANTILES)
            qmse = float(np.mean(((qr - qs) / scale) ** 2))
            per_column[name] = {"quantile_mse": qmse}
        if not per_column:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                per_column={},
                notes=[*notes, "no numeric columns"],
            )
        mean_qmse = sum(v["quantile_mse"] for v in per_column.values()) / len(per_column)
        score = float(min(1.0, max(0.0, 1.0 - np.sqrt(mean_qmse))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "mean_quantile_mse": float(mean_qmse)},
            per_column=per_column,
            notes=notes,
        )
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/__init__.py` add:

```python
from syneva.congruence import quantile_mse as quantile_mse  # side-effect: registers QuantileMSE
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"quantile_mse": "Quantile MSE",`
- `METRIC_INFO`: `"quantile_mse": "Agreement of the quantile functions of numeric columns, including the tails.",`
- `RAW_HINT`: `"quantile_mse": "Standardized mean squared error across quantiles: 0 = identical, larger = more deviation. Lower is better.",`
- `_SCALAR_LABELS`: `"mean_quantile_mse": "Mean quantile MSE",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/congruence/test_quantile_mse.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence): quantile MSE / tail accuracy (SynthEval parity)"
```

---

## Task 5: `mutual_information_difference` (Congruence, extended)

**Files:**
- Create: `src/syneva/congruence/extended/mutual_information_difference.py`
- Modify: `src/syneva/congruence/extended/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/extended/test_mutual_information_difference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/extended/test_mutual_information_difference.py
import numpy as np
import pandas as pd

from syneva.congruence.extended.mutual_information_difference import MutualInformationDifference
from syneva.core.metadata import Metadata


def test_identical_data_zero_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.1, size=400), "c": rng.normal(size=400)})
    r = MutualInformationDifference().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_mi_diff"] < 1e-9
    assert r.scalars["score"] > 0.99


def test_broken_dependency_raises_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    real = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.1, size=400)})  # a,b coupled
    syn = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})    # independent
    r = MutualInformationDifference().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_mi_diff"] > 0.1
    assert r.scalars["score"] < 0.9
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/extended/test_mutual_information_difference.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/extended/mutual_information_difference.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


def _discretize(df: pd.DataFrame, name: str, dtype: ColumnType) -> np.ndarray:
    if dtype is ColumnType.NUMERIC:
        v = pd.to_numeric(df[name], errors="coerce")
        binned = pd.qcut(v, 10, labels=False, duplicates="drop")
        return binned.fillna(-1).astype(int).to_numpy()
    return pd.factorize(df[name].astype("string").fillna("__NA__"))[0]


def _nmi_matrix(df: pd.DataFrame, meta: Metadata, cols: list[str]) -> np.ndarray:
    disc = [_discretize(df, c, meta.columns[c].dtype) for c in cols]
    k = len(cols)
    m = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = float(normalized_mutual_info_score(disc[i], disc[j]))
            m[i, j] = m[j, i] = v
    return m


@registry.register
class MutualInformationDifference:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="mutual_information_difference",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        cols = [
            n
            for n, cm in meta.columns.items()
            if cm.dtype in (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        ]
        if len(cols) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "mean_mi_diff": 0.0},
                notes=["need >=2 columns"],
            )
        mr = _nmi_matrix(real, meta, cols)
        ms = _nmi_matrix(synthetic, meta, cols)
        iu = np.triu_indices(len(cols), k=1)
        diff = float(np.mean(np.abs(mr[iu] - ms[iu])))
        score = float(min(1.0, max(0.0, 1.0 - diff)))
        return MetricResult(spec=self.spec, scalars={"score": score, "mean_mi_diff": diff})
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/extended/__init__.py` add:

```python
from syneva.congruence.extended import (
    mutual_information_difference as mutual_information_difference,  # side-effect: registers it
)
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"mutual_information_difference": "Mutual-information difference",`
- `METRIC_INFO`: `"mutual_information_difference": "Whether the pairwise dependency structure between columns is preserved.",`
- `RAW_HINT`: `"mutual_information_difference": "Mean absolute difference in normalized mutual information: 0 = identical structure, larger = more distortion. Lower is better.",`
- `_SCALAR_LABELS`: `"mean_mi_diff": "Mean MI difference",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/congruence/extended/test_mutual_information_difference.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence/extended): mutual-information matrix difference (SynthEval parity)"
```

---

## Task 6: `mmd` (Congruence, extended)

**Files:**
- Create: `src/syneva/congruence/extended/mmd.py`
- Modify: `src/syneva/congruence/extended/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/congruence/extended/test_mmd.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/congruence/extended/test_mmd.py
import numpy as np
import pandas as pd

from syneva.congruence.extended.mmd import MMD
from syneva.core.metadata import Metadata


def test_identical_distribution_small_mmd():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    r = MMD().compute(df, df, Metadata.infer(df))
    assert r.scalars["mmd"] < 0.1
    assert r.scalars["score"] > 0.9


def test_shifted_distribution_larger_mmd():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    syn = pd.DataFrame({"x": rng.normal(loc=5, size=300), "y": rng.normal(loc=5, size=300)})
    r = MMD().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mmd"] > 0.3
    assert r.scalars["score"] < 0.7
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/congruence/extended/test_mmd.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/congruence/extended/mmd.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_EPS = 1e-12
_CAP = 500


def _subsample(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(x) > _CAP:
        idx = rng.choice(len(x), _CAP, replace=False)
        return x[idx]
    return x


@registry.register
class MMD:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="mmd",
        c="congruence",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        x_real, x_syn = encode_pair(real, synthetic, meta)
        if x_real.shape[1] == 0 or len(x_real) == 0 or len(x_syn) == 0:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0, "mmd": 0.0}, notes=["no encodable columns"]
            )
        rng = np.random.default_rng(42)
        x_real = _subsample(x_real, rng)
        x_syn = _subsample(x_syn, rng)
        z = np.vstack([x_real, x_syn])
        dists = euclidean_distances(z)
        upper = dists[np.triu_indices(len(z), k=1)]
        med = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
        gamma = 1.0 / (2.0 * med**2 + _EPS)
        k_xx = rbf_kernel(x_real, x_real, gamma=gamma)
        k_yy = rbf_kernel(x_syn, x_syn, gamma=gamma)
        k_xy = rbf_kernel(x_real, x_syn, gamma=gamma)
        mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        mmd = float(np.sqrt(max(0.0, mmd2)))
        score = float(min(1.0, max(0.0, 1.0 - mmd)))
        return MetricResult(spec=self.spec, scalars={"score": score, "mmd": mmd})
```

- [ ] **Step 4: Register it**

In `src/syneva/congruence/extended/__init__.py` add:

```python
from syneva.congruence.extended import mmd as mmd  # side-effect: registers MMD
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"mmd": "Maximum mean discrepancy",`
- `METRIC_INFO`: `"mmd": "Overall multivariate distributional discrepancy between real and synthetic data, via a kernel.",`
- `RAW_HINT`: `"mmd": "0 = identical distributions, larger = more different. Lower is better.",`
- `_SCALAR_LABELS`: `"mmd": "MMD",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/congruence/extended/test_mmd.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(congruence/extended): maximum mean discrepancy (SynthEval parity)"
```

---

## Task 7: `nn_adversarial_accuracy` (Coverage, extended)

**Files:**
- Create: `src/syneva/coverage/extended/nn_adversarial_accuracy.py`
- Modify: `src/syneva/coverage/extended/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/coverage/extended/test_nn_adversarial_accuracy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/coverage/extended/test_nn_adversarial_accuracy.py
import numpy as np
import pandas as pd

from syneva.core.metadata import Metadata
from syneva.coverage.extended.nn_adversarial_accuracy import NNAdversarialAccuracy


def test_indistinguishable_data_aa_near_half():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    syn = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    r = NNAdversarialAccuracy().compute(real, syn, Metadata.infer(real))
    assert 0.4 <= r.scalars["nn_adversarial_accuracy"] <= 0.6
    assert r.scalars["score"] > 0.7


def test_separated_data_aa_near_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    syn = pd.DataFrame({"x": rng.normal(loc=20, size=400), "y": rng.normal(loc=20, size=400)})
    r = NNAdversarialAccuracy().compute(real, syn, Metadata.infer(real))
    assert r.scalars["nn_adversarial_accuracy"] > 0.9
    assert r.scalars["score"] < 0.3
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/unit/coverage/extended/test_nn_adversarial_accuracy.py --no-cov -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/coverage/extended/nn_adversarial_accuracy.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class NNAdversarialAccuracy:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="nn_adversarial_accuracy",
        c="coverage",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        x_real, x_syn = encode_pair(real, synthetic, meta)
        if x_real.shape[1] == 0 or len(x_real) < 2 or len(x_syn) < 2:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "nn_adversarial_accuracy": 0.5},
                notes=["insufficient data"],
            )
        # within-set nearest neighbour (exclude self -> take 2nd column)
        d_rr = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
        d_ss = NearestNeighbors(n_neighbors=2).fit(x_syn).kneighbors(x_syn)[0][:, 1]
        # cross-set nearest neighbour
        d_rs = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
        d_sr = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(x_syn)[0][:, 0]
        aa = 0.5 * (float(np.mean(d_rs > d_rr)) + float(np.mean(d_sr > d_ss)))
        score = float(min(1.0, max(0.0, 1.0 - 2.0 * abs(aa - 0.5))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "nn_adversarial_accuracy": float(aa)},
        )
```

- [ ] **Step 4: Register it**

In `src/syneva/coverage/extended/__init__.py` add:

```python
from syneva.coverage.extended import (
    nn_adversarial_accuracy as nn_adversarial_accuracy,  # side-effect: registers it
)
```

- [ ] **Step 5: Add metric-info entries**

- `METRIC_NAMES`: `"nn_adversarial_accuracy": "Nearest-neighbour adversarial accuracy",`
- `METRIC_INFO`: `"nn_adversarial_accuracy": "Whether synthetic points are too close (memorization) or too far (poor coverage), judged by nearest-neighbour separability.",`
- `RAW_HINT`: `"nn_adversarial_accuracy": "Adversarial accuracy from 0 to 1: 0.5 = indistinguishable (ideal), toward 1 = too separable, toward 0 = memorized. Closer to 0.5 is better.",`
- `_SCALAR_LABELS`: `"nn_adversarial_accuracy": "Adversarial accuracy",`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/coverage/extended/test_nn_adversarial_accuracy.py tests/unit/core/test_metric_info.py --no-cov -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(coverage/extended): nearest-neighbour adversarial accuracy (SynthEval parity)"
```

---

## Task 8: Parity manifest + CI-enforced coverage test

**Files:**
- Create: `tests/parity/__init__.py` (empty)
- Create: `tests/parity/syntheval_manifest.py`
- Create: `tests/parity/test_syntheval_parity.py`

- [ ] **Step 1: Write the manifest module**

```python
# tests/parity/syntheval_manifest.py
"""Mapping of SynthEval metric codes to their syneva status.

status tuple: ("implemented", syneva_metric_name) | ("planned", phase) | ("na", reason)
Flip "planned" -> "implemented" as later phases (0b/0c) land.
"""

SYNTHEVAL_PARITY: dict[str, tuple[str, str]] = {
    "corr_diff": ("implemented", "correlation_difference"),
    "ks_test": ("implemented", "ks_statistic"),
    "p_MSE": ("implemented", "pmse"),
    "fio": ("implemented", "feature_importance_spearman"),
    "cls_acc": ("implemented", "tstr_suite"),
    "auroc_diff": ("implemented", "tstr_suite"),
    "pca": ("implemented", "pca_scatter"),
    "dwm": ("implemented", "dimension_wise_means"),
    "cio": ("implemented", "ci_overlap"),
    "h_dist": ("implemented", "hellinger"),
    "q_mse": ("implemented", "quantile_mse"),
    "mi_diff": ("implemented", "mutual_information_difference"),
    "mmd": ("implemented", "mmd"),
    "nnaa": ("implemented", "nn_adversarial_accuracy"),
    "nndr": ("implemented", "nndr"),
    "dcr": ("implemented", "dcr"),
    "mia": ("implemented", "mia_auc"),
    "hit_rate": ("planned", "0b"),
    "eps_risk": ("planned", "0b"),
    "att_discl": ("planned", "0b"),
    "statistical_parity": ("planned", "0b"),
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/parity/test_syntheval_parity.py
import syneva  # noqa: F401  populate the global registry
from syneva.core.registry import registry

from tests.parity.syntheval_manifest import SYNTHEVAL_PARITY


def test_implemented_parity_metrics_are_registered():
    registered = {cls.spec.name for cls in registry.metrics()}
    missing = [
        f"{code} -> {name}"
        for code, (status, name) in SYNTHEVAL_PARITY.items()
        if status == "implemented" and name not in registered
    ]
    assert not missing, f"SynthEval-parity metrics claimed but not registered: {missing}"


def test_no_unknown_status():
    for code, (status, _) in SYNTHEVAL_PARITY.items():
        assert status in {"implemented", "planned", "na"}, f"{code} has bad status {status}"
```

- [ ] **Step 3: Run, expect pass**

Run: `uv run pytest tests/parity/ --no-cov -v`
Expected: 2 passed (all `implemented` entries from Tasks 1–7 plus the pre-existing metrics resolve to registered names; `planned` 0b entries are not checked).

Note: `tests/parity` is importable as a package because of the `__init__.py`; the import `from tests.parity.syntheval_manifest import ...` works under the repo's `importlib` pytest mode. If collection errors on the import path, run from the repo root (the default) — `tests/` is already on the path via the existing test layout.

- [ ] **Step 4: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "test(parity): CI-enforced SynthEval coverage manifest"
```

---

## Task 9: Integration + full-suite verification

**Files:**
- Modify: `tests/integration/test_full_core_scorecard.py`

- [ ] **Step 1: Add an assertion that the new metrics run end-to-end**

Append to `tests/integration/test_full_core_scorecard.py`:

```python
def test_new_fidelity_metrics_present(real_df, syn_good_df, metadata):
    import syneva

    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core", "extended"))
    names = {r.spec.name for r in rep.results}
    for m in [
        "dimension_wise_means",
        "ci_overlap",
        "hellinger",
        "quantile_mse",
        "mutual_information_difference",
        "mmd",
        "nn_adversarial_accuracy",
    ]:
        assert m in names, f"{m} not in evaluate() results"
        result = next(r for r in rep.results if r.spec.name == m)
        assert result.error is None, f"{m} errored: {result.error}"
```

- [ ] **Step 2: Run the new integration test**

Run: `uv run pytest tests/integration/test_full_core_scorecard.py --no-cov -v`
Expected: all pass (the new metrics appear and none errored).

- [ ] **Step 3: Run the FULL suite with the coverage gate**

Run: `uv run pytest -q`
Expected: all pass, coverage >= 85%.

- [ ] **Step 4: Lint + commit**

```bash
uv run ruff check . && uv run pyright src/syneva || true
git add -A
git commit -m "test(integration): new fidelity metrics run end-to-end via evaluate()"
```

(Note: `pyright` reports pre-existing strict-mode errors repo-wide and is not a commit gate here; `|| true` keeps the step from failing on that known condition.)

---

## Self-review notes

- **Spec coverage:** all 7 metrics from the spec table have a task (1–7); metric-info entries added per metric (guard tests enforce); parity manifest + test = Task 8; integration + coverage gate = Task 9. The spec's "no UI changes needed" holds — none are in the plan.
- **Type/signature consistency:** every metric uses the same `compute(self, real, synthetic, meta) -> MetricResult` signature and returns `scalars["score"]` plus its named raw scalar; metric-info keys match the scalar keys used in each `compute`; registration import names match the module filenames.
- **No placeholders:** every step contains complete, runnable code (the earlier draft note in Task 6 was removed).
