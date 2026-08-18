# C5 Preservation Dimension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `preservation` dimension — six extended-tier metrics measuring whether synthetic data preserves rare categories, distribution tails, and small subgroups — plus the `SubgroupSpec` config, runner/benchmark plumbing, and a minimal UI subgroup builder.

**Architecture:** New `src/syneva/preservation/` package mirroring `fairness/`: a `SubgroupSpec` dataclass with a shared `matches()`/`validate()` row-selection implementation, one module per metric registered via `@registry.register`, shared column-type helpers in `_util.py`. The runner gains a `subgroup_specs` parameter and a drop rule (metrics whose constructor wants `subgroup_specs` are not instantiated when none are provided). All metrics are bespoke lightweight statistics — no model panels, no re-use of other dimensions' metric classes.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (`ks_2samp`), scikit-learn (LogisticRegression pipeline, NearestNeighbors), pytest. Spec: `docs/superpowers/specs/2026-08-18-syneva-preservation-5.md`.

## Global Constraints

- Branch: `feat/preservation-5` (created in Task 1, all commits land there).
- TDD: every metric's tests are written and observed failing before implementation.
- Run tests with `uv run pytest` (never bare `pytest`).
- NEVER pass `--update-golden`. The golden snapshot must not change (all new metrics are `tier="extended"`; golden runs `tiers=("core",)`).
- Coverage gate: `--cov-fail-under=85` is enforced by `pyproject.toml`; new source files must be well covered.
- Every registered metric MUST have entries in `METRIC_NAMES`, `METRIC_INFO` (neutral — the guard test rejects "higher is better"/"lower is better" phrasing), and `RAW_HINT` in `src/syneva/core/metric_info.py`, in the same task that registers the metric — otherwise `tests/unit/core/test_metric_info.py` fails.
- All metric specs: `c="preservation"`, `tier="extended"`, `data_types=frozenset({"static"})`, `requires_real=True`, `scope="table-level"`. Every `score` scalar is in [0, 1], 1 = ideal.
- Defaults fixed by the spec: `rare_threshold=0.05`, `tail_quantile=0.05`, `min_rows=10`, `cap=2000`, `random_state=42`, `_EPS = 1e-12`.
- "Categorical columns" = `ColumnType.CATEGORICAL` **and** `ColumnType.BOOLEAN`; "numeric columns" = `ColumnType.NUMERIC` only; ID/DATETIME columns are ignored by every preservation metric.

---

### Task 1: Package skeleton, `SubgroupSpec`, C Literal, C_INFO

**Files:**
- Create: `src/syneva/preservation/__init__.py`
- Create: `src/syneva/preservation/spec.py`
- Create: `src/syneva/preservation/_util.py`
- Modify: `src/syneva/core/metric.py` (C Literal, lines 12–22)
- Modify: `src/syneva/core/metric_info.py` (C_INFO, lines 18–24)
- Modify: `src/syneva/__init__.py` (re-export + side-effect import)
- Test: `tests/unit/preservation/__init__.py`, `tests/unit/preservation/test_spec.py`

**Interfaces:**
- Consumes: `Metadata`, `ColumnType` from `syneva.core.metadata`.
- Produces (later tasks rely on these exact names):
  - `SubgroupSpec(name: str, conditions: dict[str, list | tuple])` with
    `matches(df: pd.DataFrame) -> pd.Series` (boolean mask, df.index-aligned) and
    `validate(meta: Metadata) -> str | None` (None = valid, else problem text).
  - `syneva.preservation._util.categorical_columns(meta) -> list[str]`
  - `syneva.preservation._util.numeric_columns(meta) -> list[str]`
  - `syneva.preservation._util.runnable_specs(specs, meta) -> tuple[list[SubgroupSpec], list[str]]`
    (valid specs + one note per rejected spec).
  - `SubgroupSpec` importable as `from syneva import SubgroupSpec`.

- [ ] **Step 1: Create branch**

```bash
cd /Users/changsun/Documents/Syn_Eva && git checkout -b feat/preservation-5
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/preservation/__init__.py`: empty file.

`tests/unit/preservation/test_spec.py`:

```python
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric_info import describe_c
from syneva.preservation._util import categorical_columns, numeric_columns, runnable_specs


def _meta():
    return Metadata(
        columns={
            "sex": ColumnMetadata(name="sex", dtype=ColumnType.CATEGORICAL),
            "age": ColumnMetadata(name="age", dtype=ColumnType.NUMERIC),
            "flag": ColumnMetadata(name="flag", dtype=ColumnType.BOOLEAN),
            "row_id": ColumnMetadata(name="row_id", dtype=ColumnType.ID),
        }
    )


def _df():
    return pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M", "F"],
            "age": [70, 30, 66, 20, 64],
            "flag": [True, False, True, True, False],
            "row_id": [1, 2, 3, 4, 5],
        }
    )


def test_subgroup_spec_fields():
    sp = SubgroupSpec(name="elderly women", conditions={"sex": ["F"], "age": (65, None)})
    assert sp.name == "elderly women"
    assert sp.conditions["age"] == (65, None)


def test_matches_values_and_range():
    sp = SubgroupSpec(name="ew", conditions={"sex": ["F"], "age": (65, None)})
    mask = sp.matches(_df())
    assert mask.tolist() == [True, False, True, False, False]


def test_matches_range_bounds_inclusive():
    sp = SubgroupSpec(name="band", conditions={"age": (30, 66)})
    assert sp.matches(_df()).tolist() == [False, True, True, False, True]


def test_matches_open_lower_bound():
    sp = SubgroupSpec(name="young", conditions={"age": (None, 30)})
    assert sp.matches(_df()).tolist() == [False, True, False, True, False]


def test_validate_ok():
    sp = SubgroupSpec(name="ok", conditions={"sex": ["F"], "age": (None, 30)})
    assert sp.validate(_meta()) is None


def test_validate_unknown_column():
    sp = SubgroupSpec(name="bad", conditions={"nope": ["x"]})
    assert "unknown column" in sp.validate(_meta())


def test_validate_range_on_categorical():
    sp = SubgroupSpec(name="bad", conditions={"sex": (0, 1)})
    assert "range" in sp.validate(_meta())


def test_validate_values_on_numeric():
    sp = SubgroupSpec(name="bad", conditions={"age": [65]})
    assert "values list" in sp.validate(_meta())


def test_util_column_selectors():
    meta = _meta()
    assert categorical_columns(meta) == ["sex", "flag"]
    assert numeric_columns(meta) == ["age"]


def test_runnable_specs_partitions():
    good = SubgroupSpec(name="g", conditions={"sex": ["F"]})
    bad = SubgroupSpec(name="b", conditions={"nope": ["x"]})
    ok, notes = runnable_specs([good, bad], _meta())
    assert ok == [good]
    assert len(notes) == 1 and "b" in notes[0]


def test_preservation_dimension_described():
    assert describe_c("preservation") != ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_spec.py -v`
Expected: FAIL with `ImportError: cannot import name 'SubgroupSpec' from 'syneva'`

- [ ] **Step 4: Implement**

`src/syneva/preservation/spec.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from syneva.core.metadata import ColumnType, Metadata


@dataclass
class SubgroupSpec:
    """Declares a subgroup of rows for the preservation metrics: a display name
    and AND-ed per-column conditions — a list of allowed values for a
    categorical/boolean column, or an inclusive ``(lo, hi)`` range tuple for a
    numeric column (``None`` = open end)."""

    name: str
    conditions: dict[str, list | tuple]

    def validate(self, meta: Metadata) -> str | None:
        """Return a problem description, or None when the spec is runnable."""
        for col, cond in self.conditions.items():
            if col not in meta.columns:
                return f"unknown column '{col}'"
            dtype = meta.columns[col].dtype
            if isinstance(cond, tuple):
                if dtype is not ColumnType.NUMERIC:
                    return f"range condition on non-numeric column '{col}'"
                if len(cond) != 2:
                    return f"range for '{col}' must be a (lo, hi) tuple"
            elif isinstance(cond, list):
                if dtype is ColumnType.NUMERIC:
                    return f"values list on numeric column '{col}'"
            else:
                return f"condition for '{col}' must be a list or a (lo, hi) tuple"
        return None

    def matches(self, df: pd.DataFrame) -> pd.Series:
        """Boolean mask (aligned to df.index) of rows in the subgroup."""
        mask = pd.Series(True, index=df.index)
        for col, cond in self.conditions.items():
            if isinstance(cond, tuple):
                v = pd.to_numeric(df[col], errors="coerce")
                lo, hi = cond
                if lo is not None:
                    mask &= v >= lo
                if hi is not None:
                    mask &= v <= hi
            else:
                mask &= df[col].isin(cond)
        return mask
```

`src/syneva/preservation/_util.py`:

```python
from __future__ import annotations

from syneva.core.metadata import ColumnType, Metadata
from syneva.preservation.spec import SubgroupSpec

_EPS = 1e-12


def categorical_columns(meta: Metadata) -> list[str]:
    return [
        n
        for n, cm in meta.columns.items()
        if cm.dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]


def numeric_columns(meta: Metadata) -> list[str]:
    return [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]


def runnable_specs(
    specs: list[SubgroupSpec], meta: Metadata
) -> tuple[list[SubgroupSpec], list[str]]:
    """Split specs into runnable ones and a note per rejected spec."""
    ok: list[SubgroupSpec] = []
    notes: list[str] = []
    for sp in specs:
        problem = sp.validate(meta)
        if problem is None:
            ok.append(sp)
        else:
            notes.append(f"subgroup '{sp.name}' skipped ({problem})")
    return ok, notes
```

`src/syneva/preservation/__init__.py` (metric imports appended by later tasks):

```python
from syneva.preservation.spec import SubgroupSpec as SubgroupSpec
```

`src/syneva/core/metric.py` — add to the `C` Literal after `"fairness",`:

```python
    "preservation",
```

`src/syneva/core/metric_info.py` — add to `C_INFO` after the `"fairness"` entry:

```python
    "preservation": "Does the synthetic data preserve rare categories, distribution tails, and small subgroups of the real data?",
```

`src/syneva/__init__.py` — add `from syneva.preservation.spec import SubgroupSpec` after the `FairnessSpec` import, add `"SubgroupSpec"` to `__all__` (keep alphabetical order), and append the side-effect import at the bottom with the others:

```python
from syneva import preservation as _preservation  # noqa: F401
```

- [ ] **Step 5: Run tests to verify they pass, then commit**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

```bash
git add src/syneva/preservation src/syneva/core/metric.py src/syneva/core/metric_info.py src/syneva/__init__.py tests/unit/preservation
git commit -m "feat(preservation): SubgroupSpec, dimension registration, package skeleton"
```

---

### Task 2: `rare_category_retention`

**Files:**
- Create: `src/syneva/preservation/rare_category_retention.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_rare_category_retention.py`

**Interfaces:**
- Consumes: `categorical_columns` from Task 1; `MetricSpec`/`MetricResult`/`registry` (existing core).
- Produces: registered metric `rare_category_retention`, constructor `RareCategoryRetention(rare_threshold: float = 0.05)`, scalars `{score, n_rare_categories, pct_rare_missing}`, `per_column[col] = {"retention": float}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_rare_category_retention.py`:

```python
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.rare_category_retention import RareCategoryRetention


def _meta():
    return Metadata(columns={"cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL)})


def _real():
    # "B" has frequency 4% < 5% threshold -> rare
    return pd.DataFrame({"cat": ["A"] * 96 + ["B"] * 4})


def test_rare_category_kept_scores_one():
    res = RareCategoryRetention().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert res.scalars["n_rare_categories"] == 1.0
    assert res.scalars["pct_rare_missing"] == 0.0


def test_rare_category_dropped_scores_zero():
    syn = pd.DataFrame({"cat": ["A"] * 100})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["pct_rare_missing"] == 1.0


def test_rare_category_halved_scores_half():
    syn = pd.DataFrame({"cat": ["A"] * 98 + ["B"] * 2})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 1e-9


def test_oversampled_rare_category_capped_at_one():
    syn = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    res = RareCategoryRetention().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 1.0


def test_no_rare_categories_scores_one_with_note():
    real = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
    res = RareCategoryRetention().compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("no rare categories" in n for n in res.notes)


def test_no_categorical_columns_scores_one_with_note():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    res = RareCategoryRetention().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert res.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_rare_category_retention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syneva.preservation.rare_category_retention'`

- [ ] **Step 3: Implement**

`src/syneva/preservation/rare_category_retention.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns


@registry.register
@dataclass
class RareCategoryRetention:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="rare_category_retention",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    rare_threshold: float = 0.05

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        retentions: list[float] = []
        n_missing = 0
        cat_cols = categorical_columns(meta)
        if not cat_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no categorical columns"]
            )
        for name in cat_cols:
            p_real = real[name].value_counts(normalize=True)
            p_syn = synthetic[name].value_counts(normalize=True)
            rare = p_real[p_real < self.rare_threshold]
            if rare.empty:
                continue
            col_retentions: list[float] = []
            for cat, pr in rare.items():
                ps = float(p_syn.get(cat, 0.0))
                if ps == 0.0:
                    n_missing += 1
                col_retentions.append(min(ps / float(pr), 1.0))
            retentions.extend(col_retentions)
            per_column[name] = {"retention": sum(col_retentions) / len(col_retentions)}
        if not retentions:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no rare categories in the real data"],
            )
        score = float(min(1.0, max(0.0, sum(retentions) / len(retentions))))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "n_rare_categories": float(len(retentions)),
                "pct_rare_missing": n_missing / len(retentions),
            },
            per_column=per_column,
            notes=notes,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    rare_category_retention as rare_category_retention,  # side-effect: registers metric
)
```

Add to `src/syneva/core/metric_info.py` — in `METRIC_INFO` (new `# Preservation` comment block after the fairness entry):

```python
    # Preservation — minorities, tails, and subgroups
    "rare_category_retention": "Whether categories that are rare in the real data survive into the synthetic data.",
```

In `RAW_HINT`:

```python
    "rare_category_retention": "Mean retention of rare categories, 0 to 1: 1 = every rare category keeps its real frequency, 0 = all rare categories lost. Higher is better.",
```

In `METRIC_NAMES`:

```python
    "rare_category_retention": "Rare-category retention",
```

In `_SCALAR_LABELS`:

```python
    "n_rare_categories": "Rare categories found",
    "pct_rare_missing": "Share of rare categories missing",
    "retention": "Mean retention",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS (metric-info guards now cover the new metric)

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): rare_category_retention metric"
```

---

### Task 3: `tail_coverage`

**Files:**
- Create: `src/syneva/preservation/tail_coverage.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_tail_coverage.py`

**Interfaces:**
- Consumes: `numeric_columns` from Task 1.
- Produces: registered metric `tail_coverage`, constructor `TailCoverage(tail_quantile: float = 0.05)`, scalars `{score, lower_tail_coverage, upper_tail_coverage}`, `per_column[col] = {"tail_coverage": float}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_tail_coverage.py`:

```python
import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.tail_coverage import TailCoverage


def _meta():
    return Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})


def _real():
    return pd.DataFrame({"x": np.linspace(0.0, 1.0, 1001)})


def test_identical_data_covers_tails():
    res = TailCoverage().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] > 0.9


def test_truncated_synthetic_scores_zero():
    syn = pd.DataFrame({"x": np.linspace(0.1, 0.9, 1001)})
    res = TailCoverage().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["lower_tail_coverage"] == 0.0
    assert res.scalars["upper_tail_coverage"] == 0.0


def test_one_sided_truncation():
    syn = pd.DataFrame({"x": np.linspace(0.0, 0.9, 1001)})
    res = TailCoverage().compute(_real(), syn, _meta())
    assert res.scalars["upper_tail_coverage"] == 0.0
    assert res.scalars["lower_tail_coverage"] > 0.9


def test_constant_column_skipped_with_note():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0] * 100})
    res = TailCoverage().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert any("degenerate" in n for n in res.notes)


def test_no_numeric_columns_scores_one_with_note():
    meta = Metadata(columns={"c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL)})
    df = pd.DataFrame({"c": ["a", "b"]})
    res = TailCoverage().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert res.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_tail_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/syneva/preservation/tail_coverage.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import numeric_columns


@registry.register
@dataclass
class TailCoverage:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="tail_coverage",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tail_quantile: float = 0.05

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        notes: list[str] = []
        per_column: dict[str, dict[str, float]] = {}
        lowers: list[float] = []
        uppers: list[float] = []
        num_cols = numeric_columns(meta)
        if not num_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no numeric columns"]
            )
        q = self.tail_quantile
        for name in num_cols:
            rv = pd.to_numeric(real[name], errors="coerce").dropna()
            sv = pd.to_numeric(synthetic[name], errors="coerce").dropna()
            if rv.empty or sv.empty:
                notes.append(f"column '{name}' skipped (no numeric values)")
                continue
            lo, hi = float(rv.quantile(q)), float(rv.quantile(1 - q))
            if lo == hi:
                notes.append(f"column '{name}' skipped (degenerate quantiles)")
                continue
            lower = min(float((sv < lo).mean()) / q, 1.0)
            upper = min(float((sv > hi).mean()) / q, 1.0)
            lowers.append(lower)
            uppers.append(upper)
            per_column[name] = {"tail_coverage": (lower + upper) / 2}
        if not lowers:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no usable numeric columns"],
            )
        mean_lower = sum(lowers) / len(lowers)
        mean_upper = sum(uppers) / len(uppers)
        score = float(min(1.0, max(0.0, (mean_lower + mean_upper) / 2)))
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "lower_tail_coverage": mean_lower,
                "upper_tail_coverage": mean_upper,
            },
            per_column=per_column,
            notes=notes,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    tail_coverage as tail_coverage,  # side-effect: registers metric
)
```

`src/syneva/core/metric_info.py` — `METRIC_INFO`:

```python
    "tail_coverage": "Whether synthetic values reach into the extreme tails of each numeric column.",
```

`RAW_HINT`:

```python
    "tail_coverage": "Fraction of the expected tail mass the synthetic data reproduces, 0 to 1: 1 = both tails fully populated, 0 = tails empty. Higher is better.",
```

`METRIC_NAMES`:

```python
    "tail_coverage": "Tail coverage",
```

`_SCALAR_LABELS`:

```python
    "lower_tail_coverage": "Lower-tail coverage",
    "upper_tail_coverage": "Upper-tail coverage",
    "tail_coverage": "Tail coverage",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): tail_coverage metric"
```

---

### Task 4: `minority_class_density`

**Files:**
- Create: `src/syneva/preservation/minority_class_density.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_minority_class_density.py`

**Interfaces:**
- Consumes: `categorical_columns` from Task 1.
- Produces: registered metric `minority_class_density`, constructor `MinorityClassDensity()` (no config), scalars `{score, worst_density_ratio}`, `per_column[col] = {"density_ratio": float}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_minority_class_density.py`:

```python
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_class_density import MinorityClassDensity


def _meta():
    return Metadata(columns={"cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL)})


def _real():
    # minority class "B" at 10%
    return pd.DataFrame({"cat": ["A"] * 90 + ["B"] * 10})


def test_identical_scores_one():
    res = MinorityClassDensity().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0


def test_halved_minority_scores_about_half():
    syn = pd.DataFrame({"cat": ["A"] * 95 + ["B"] * 5})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 0.02


def test_doubled_minority_also_penalized():
    syn = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert abs(res.scalars["score"] - 0.5) < 0.02


def test_vanished_minority_scores_zero():
    syn = pd.DataFrame({"cat": ["A"] * 100})
    res = MinorityClassDensity().compute(_real(), syn, _meta())
    assert res.scalars["score"] == 0.0
    assert res.scalars["worst_density_ratio"] == 0.0


def test_minority_tie_is_deterministic():
    real = pd.DataFrame({"cat": ["A"] * 40 + ["B"] * 30 + ["C"] * 30})
    syn = pd.DataFrame({"cat": ["A"] * 40 + ["B"] * 15 + ["C"] * 45})
    # tie between B and C -> sorted label "B" wins; B halved -> ratio 0.5
    res = MinorityClassDensity().compute(real, syn, _meta())
    assert abs(res.per_column["cat"]["density_ratio"] - 0.5) < 0.02


def test_no_categorical_columns_scores_one_with_note():
    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"x": [1.0, 2.0]})
    res = MinorityClassDensity().compute(df, df.copy(), meta)
    assert res.scalars["score"] == 1.0
    assert res.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_minority_class_density.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/syneva/preservation/minority_class_density.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns


@registry.register
@dataclass
class MinorityClassDensity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_class_density",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        ratios: list[float] = []
        cat_cols = categorical_columns(meta)
        if not cat_cols:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no categorical columns"]
            )
        for name in cat_cols:
            p_real = real[name].value_counts(normalize=True)
            if p_real.empty:
                continue
            # least-frequent class; ties broken by sorted label for determinism
            minority = min(p_real.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
            pr = float(p_real[minority])
            ps = float(synthetic[name].value_counts(normalize=True).get(minority, 0.0))
            ratio = 0.0 if max(pr, ps) == 0 else min(pr, ps) / max(pr, ps)
            ratios.append(ratio)
            per_column[name] = {"density_ratio": ratio}
        if not ratios:
            return MetricResult(
                spec=self.spec, scalars={"score": 1.0}, notes=["no usable categorical columns"]
            )
        score = float(min(1.0, max(0.0, sum(ratios) / len(ratios))))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "worst_density_ratio": min(ratios)},
            per_column=per_column,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    minority_class_density as minority_class_density,  # side-effect: registers metric
)
```

`src/syneva/core/metric_info.py` — `METRIC_INFO`:

```python
    "minority_class_density": "Whether the least-frequent class of each categorical column keeps its share of the data.",
```

`RAW_HINT`:

```python
    "minority_class_density": "Symmetric density ratio of each column's least-frequent class, 0 to 1: 1 = share preserved, 0 = class vanished; over-representation is penalized the same as under-representation. Higher is better.",
```

`METRIC_NAMES`:

```python
    "minority_class_density": "Minority-class density",
```

`_SCALAR_LABELS`:

```python
    "worst_density_ratio": "Worst density ratio",
    "density_ratio": "Density ratio",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): minority_class_density metric"
```

---

### Task 5: `subgroup_fidelity`

**Files:**
- Create: `src/syneva/preservation/subgroup_fidelity.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_subgroup_fidelity.py`

**Interfaces:**
- Consumes: `SubgroupSpec`, `runnable_specs`, `categorical_columns`, `numeric_columns` from Task 1; `scipy.stats.ks_2samp`.
- Produces: registered metric `subgroup_fidelity`, constructor `SubgroupFidelity(subgroup_specs: list[SubgroupSpec] = [], min_rows: int = 10)`. The **`subgroup_specs` constructor param name is the runner-injection contract** (Task 8). Scalars `{score, worst_subgroup_score, mean_share_drift}`, `per_column[spec.name] = {share_real, share_syn, shape_score, score}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_subgroup_fidelity.py`:

```python
import numpy as np
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.subgroup_fidelity import SubgroupFidelity


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def _real():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"g": ["A"] * 160 + ["B"] * 40, "x": np.concatenate([rng.normal(0, 1, 160), rng.normal(5, 1, 40)])}
    )


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})


def test_identical_scores_high():
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] > 0.95
    assert res.per_column["b-group"]["share_real"] == 0.2


def test_erased_subgroup_scores_zero():
    syn = pd.DataFrame({"g": ["A"] * 200, "x": np.zeros(200)})
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(_real(), syn, _meta())
    assert res.per_column["b-group"]["score"] == 0.0
    assert res.scalars["score"] == 0.0
    assert any("erased" in n for n in res.notes)


def test_shrunk_subgroup_drops_share_term():
    real = _real()
    syn = pd.concat([real[real.g == "A"], real[real.g == "B"].head(20)], ignore_index=True)
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    # share_syn ~ 20/180 = 0.111 vs share_real 0.2 -> share term ~0.556; shape
    # stays high (same-distribution subsample) but KS noise on n=20 is real
    assert 0.6 < res.scalars["score"] < 0.9


def test_shifted_subgroup_drops_shape_term():
    real = _real()
    syn = real.copy()
    syn.loc[syn.g == "B", "x"] = syn.loc[syn.g == "B", "x"] + 10
    res = SubgroupFidelity(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.per_column["b-group"]["shape_score"] < 0.6
    assert res.per_column["b-group"]["share_real"] == 0.2


def test_tiny_real_subgroup_skipped():
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    res = SubgroupFidelity(subgroup_specs=[spec]).compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("tiny" in n for n in res.notes)


def test_invalid_spec_skipped_with_note():
    spec = SubgroupSpec(name="bad", conditions={"g": (0, 1)})
    res = SubgroupFidelity(subgroup_specs=[spec]).compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("bad" in n for n in res.notes)


def test_no_specs_scores_one_with_note():
    res = SubgroupFidelity().compute(_real(), _real().copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert res.notes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_subgroup_fidelity.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/syneva/preservation/subgroup_fidelity.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from scipy.stats import ks_2samp

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import categorical_columns, numeric_columns, runnable_specs
from syneva.preservation.spec import SubgroupSpec


def _column_distances(
    real_sub: pd.DataFrame, syn_sub: pd.DataFrame, meta
) -> list[float]:
    """Per-column distributional distances within the subgroup, each in [0, 1]:
    two-sample KS statistic for numeric columns, total-variation distance for
    categorical/boolean columns."""
    dists: list[float] = []
    for name in numeric_columns(meta):
        rv = pd.to_numeric(real_sub[name], errors="coerce").dropna()
        sv = pd.to_numeric(syn_sub[name], errors="coerce").dropna()
        if rv.empty or sv.empty:
            continue
        dists.append(float(ks_2samp(rv, sv).statistic))
    for name in categorical_columns(meta):
        pr = real_sub[name].value_counts(normalize=True)
        ps = syn_sub[name].value_counts(normalize=True)
        cats = set(pr.index) | set(ps.index)
        if not cats:
            continue
        dists.append(0.5 * sum(abs(float(pr.get(c, 0.0)) - float(ps.get(c, 0.0))) for c in cats))
    return dists


@registry.register
@dataclass
class SubgroupFidelity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="subgroup_fidelity",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    min_rows: int = 10

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        per_column: dict[str, dict[str, float]] = {}
        scores: list[float] = []
        share_drifts: list[float] = []
        for sp in specs:
            rmask = sp.matches(real)
            if int(rmask.sum()) < self.min_rows:
                notes.append(f"subgroup '{sp.name}' skipped (<{self.min_rows} real rows)")
                continue
            smask = sp.matches(synthetic)
            share_real = float(rmask.mean())
            share_syn = float(smask.mean())
            share_drifts.append(abs(share_syn - share_real))
            if int(smask.sum()) == 0:
                notes.append(f"subgroup '{sp.name}' erased in the synthetic data")
                per_column[sp.name] = {
                    "share_real": share_real,
                    "share_syn": 0.0,
                    "shape_score": 0.0,
                    "score": 0.0,
                }
                scores.append(0.0)
                continue
            share_score = min(share_syn, share_real) / max(share_syn, share_real)
            dists = _column_distances(real[rmask], synthetic[smask], meta)
            shape_score = 1.0 - (sum(dists) / len(dists)) if dists else 1.0
            score_spec = float(min(1.0, max(0.0, 0.5 * shape_score + 0.5 * share_score)))
            per_column[sp.name] = {
                "share_real": share_real,
                "share_syn": share_syn,
                "shape_score": shape_score,
                "score": score_spec,
            }
            scores.append(score_spec)
        if not scores:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable subgroup specs"],
            )
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float(sum(scores) / len(scores)),
                "worst_subgroup_score": min(scores),
                "mean_share_drift": sum(share_drifts) / len(share_drifts),
            },
            per_column=per_column,
            notes=notes,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    subgroup_fidelity as subgroup_fidelity,  # side-effect: registers metric
)
```

`src/syneva/core/metric_info.py` — `METRIC_INFO`:

```python
    "subgroup_fidelity": "Whether declared subgroups keep their size and their internal distributions in the synthetic data.",
```

`RAW_HINT`:

```python
    "subgroup_fidelity": "Even blend of subgroup-size preservation and within-subgroup distribution match, 0 to 1: 1 = subgroup fully preserved, 0 = subgroup erased. Higher is better.",
```

`METRIC_NAMES`:

```python
    "subgroup_fidelity": "Subgroup fidelity",
```

`_SCALAR_LABELS`:

```python
    "worst_subgroup_score": "Worst subgroup score",
    "mean_share_drift": "Mean subgroup-share drift",
    "share_real": "Subgroup share (real)",
    "share_syn": "Subgroup share (synthetic)",
    "shape_score": "Within-subgroup shape score",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): subgroup_fidelity metric"
```

---

### Task 6: `minority_utility_gap`

**Files:**
- Create: `src/syneva/preservation/minority_utility_gap.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_minority_utility_gap.py`

**Interfaces:**
- Consumes: `SubgroupSpec`, `runnable_specs` from Task 1; `UtilityTask` from `syneva.utility.task`; sklearn.
- Produces: registered metric `minority_utility_gap`, constructor
  `MinorityUtilityGap(subgroup_specs=[], tasks=None, holdout=None, min_rows=10, random_state=42)`.
  The `tasks` and `holdout` param names bind to the runner's existing injection; `tasks` may arrive
  as `None` (when `run_utility=False` the runner never auto-suggests) — treated as "no tasks".
  Scalars `{score, worst_excess_gap, mean_gap_synthetic, mean_gap_real}`,
  `per_column[f"{spec.name}|{task.target}"] = {gap_synthetic, gap_real, excess_gap}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_minority_utility_gap.py`:

```python
import numpy as np
import pandas as pd

from syneva import SubgroupSpec, UtilityTask
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_utility_gap import MinorityUtilityGap


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def _real():
    # y = 1[x > 0] for group A, y = 1[x > 5] for group B: group-shifted
    # thresholds a linear model WITH group dummies can represent exactly.
    # (Do NOT test with subgroup-flipped labels: a no-interaction logistic
    # model cannot learn a flip, so the distortion never reaches the eval.)
    n_a, n_b = 280, 120
    xa = np.tile([-1.0, 1.0], n_a // 2)
    xb = np.tile([4.0, 6.0], n_b // 2)
    x = np.concatenate([xa, xb])
    y = np.concatenate([(xa > 0), (xb > 5)]).astype(int).astype(str)
    g = ["A"] * n_a + ["B"] * n_b
    return pd.DataFrame({"g": g, "x": x, "y": y})


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})
_TASK = UtilityTask(target="y", task_type="classification")


def test_faithful_synthetic_scores_one():
    real = _real()
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK])
    res = m.compute(real, real.copy(), _meta())
    assert res.scalars["score"] > 0.95


def test_distorted_subgroup_relationship_scores_low():
    real = _real()
    syn = real.copy()
    b = syn.g == "B"
    # shift B's x by -4: B's learned threshold moves from 5 to 1, so the
    # synthetic-trained model misclassifies real B rows at x=4
    syn.loc[b, "x"] = syn.loc[b, "x"] - 4.0
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK])
    res = m.compute(real, syn, _meta())
    assert res.scalars["score"] < 0.85
    assert res.per_column["b-group|y"]["excess_gap"] > 0.1


def test_no_tasks_scores_one_with_note():
    real = _real()
    res = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=None).compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("task" in n for n in res.notes)


def test_regression_tasks_skipped():
    real = _real()
    task = UtilityTask(target="x", task_type="regression")
    res = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[task]).compute(
        real, real.copy(), _meta()
    )
    assert res.scalars["score"] == 1.0
    assert any("regression" in n for n in res.notes)


def test_tiny_subgroup_pair_skipped():
    real = _real()
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    res = MinorityUtilityGap(subgroup_specs=[spec], tasks=[_TASK]).compute(
        real, real.copy(), _meta()
    )
    assert res.scalars["score"] == 1.0
    assert any("tiny" in n for n in res.notes)


def test_holdout_used_as_eval_frame():
    real = _real()
    holdout = _real()
    m = MinorityUtilityGap(subgroup_specs=[_SPEC], tasks=[_TASK], holdout=holdout)
    res = m.compute(real, real.copy(), _meta())
    assert res.scalars["score"] > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_minority_utility_gap.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/syneva/preservation/minority_utility_gap.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from syneva.core.metadata import ColumnType
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import runnable_specs
from syneva.preservation.spec import SubgroupSpec
from syneva.utility.task import UtilityTask


def _make_model(meta, features: list[str]) -> Pipeline:
    num = [f for f in features if meta.columns[f].dtype is ColumnType.NUMERIC]
    cat = [f for f in features if meta.columns[f].dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


@registry.register
@dataclass
class MinorityUtilityGap:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_utility_gap",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    tasks: list[UtilityTask] | None = None
    holdout: pd.DataFrame | None = None
    min_rows: int = 10
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        if not self.tasks:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no utility tasks configured; utility gap not computed"],
            )
        class_tasks = []
        for t in self.tasks:
            kind = t.task_type
            if kind is None:
                kind = (
                    "classification"
                    if meta.columns[t.target].dtype
                    in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
                    else "regression"
                )
            if kind == "classification":
                class_tasks.append(t)
            else:
                notes.append(f"task '{t.target}' skipped (regression not supported)")
        if not class_tasks or not specs:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable (task x subgroup) pairs"],
            )

        if self.holdout is not None:
            train_real, eval_real = real, self.holdout
        else:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(len(real))
            half = len(real) // 2
            train_real = real.iloc[perm[:half]].reset_index(drop=True)
            eval_real = real.iloc[perm[half:]].reset_index(drop=True)

        per_column: dict[str, dict[str, float]] = {}
        pair_scores: list[float] = []
        gaps_syn: list[float] = []
        gaps_real: list[float] = []
        excesses: list[float] = []
        for t in class_tasks:
            features = t.features or [
                n
                for n, cm in meta.columns.items()
                if n != t.target and cm.dtype is not ColumnType.ID
            ]
            y_syn = synthetic[t.target].astype(str)
            y_train_real = train_real[t.target].astype(str)
            y_eval = eval_real[t.target].astype(str)
            if y_syn.nunique() < 2 or y_train_real.nunique() < 2:
                notes.append(f"task '{t.target}' skipped (constant target in training data)")
                continue
            model_syn = _make_model(meta, features).fit(synthetic[features], y_syn)
            model_real = _make_model(meta, features).fit(train_real[features], y_train_real)
            pred_syn = model_syn.predict(eval_real[features])
            pred_real = model_real.predict(eval_real[features])
            overall_syn = balanced_accuracy_score(y_eval, pred_syn)
            overall_real = balanced_accuracy_score(y_eval, pred_real)
            for sp in specs:
                mask = sp.matches(eval_real).to_numpy()
                if int(mask.sum()) < self.min_rows:
                    notes.append(
                        f"pair '{sp.name}|{t.target}' skipped (<{self.min_rows} eval rows)"
                    )
                    continue
                if y_eval[mask].nunique() < 2:
                    notes.append(
                        f"pair '{sp.name}|{t.target}' skipped (constant target in subgroup)"
                    )
                    continue
                sub_syn = balanced_accuracy_score(y_eval[mask], pred_syn[mask])
                sub_real = balanced_accuracy_score(y_eval[mask], pred_real[mask])
                gap_syn = float(overall_syn - sub_syn)
                gap_real = float(overall_real - sub_real)
                excess = max(0.0, gap_syn - gap_real)
                per_column[f"{sp.name}|{t.target}"] = {
                    "gap_synthetic": gap_syn,
                    "gap_real": gap_real,
                    "excess_gap": excess,
                }
                gaps_syn.append(gap_syn)
                gaps_real.append(gap_real)
                excesses.append(excess)
                pair_scores.append(float(min(1.0, max(0.0, 1.0 - excess))))
        if not pair_scores:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable (task x subgroup) pairs"],
            )
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": float(sum(pair_scores) / len(pair_scores)),
                "worst_excess_gap": max(excesses),
                "mean_gap_synthetic": sum(gaps_syn) / len(gaps_syn),
                "mean_gap_real": sum(gaps_real) / len(gaps_real),
            },
            per_column=per_column,
            notes=notes,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    minority_utility_gap as minority_utility_gap,  # side-effect: registers metric
)
```

`src/syneva/core/metric_info.py` — `METRIC_INFO`:

```python
    "minority_utility_gap": "Whether a model trained on synthetic data serves declared subgroups as well as it serves the overall population.",
```

`RAW_HINT`:

```python
    "minority_utility_gap": "Excess gap = synthetic-trained performance gap minus real-trained gap, floored at 0. 0 means training on synthetic data costs the subgroup nothing beyond what real data already would. Lower is better.",
```

`METRIC_NAMES`:

```python
    "minority_utility_gap": "Minority utility gap",
```

`_SCALAR_LABELS`:

```python
    "worst_excess_gap": "Worst excess gap",
    "mean_gap_synthetic": "Mean gap (synthetic-trained)",
    "mean_gap_real": "Mean gap (real-trained)",
    "gap_synthetic": "Gap (synthetic-trained)",
    "gap_real": "Gap (real-trained)",
    "excess_gap": "Excess gap",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): minority_utility_gap metric"
```

---

### Task 7: `minority_privacy_risk`

**Files:**
- Create: `src/syneva/preservation/minority_privacy_risk.py`
- Modify: `src/syneva/preservation/__init__.py`
- Modify: `src/syneva/core/metric_info.py`
- Test: `tests/unit/preservation/test_minority_privacy_risk.py`

**Interfaces:**
- Consumes: `SubgroupSpec`, `runnable_specs` from Task 1; `encode_pair` from `syneva.compliance._encode`; `gower_matrix` from `syneva.core.distance`; sklearn `NearestNeighbors`.
- Produces: registered metric `minority_privacy_risk`, constructor
  `MinorityPrivacyRisk(subgroup_specs=[], distance="euclidean", min_rows=10, cap=2000, random_state=42)`
  (`distance` binds to the runner's existing injection). Scalars
  `{score, mean_risk_ratio, worst_risk_ratio}`,
  `per_column[spec.name] = {risk_ratio, median_dcr_subgroup, median_dcr_overall}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/preservation/test_minority_privacy_risk.py`:

```python
import numpy as np
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.preservation.minority_privacy_risk import MinorityPrivacyRisk


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def _real():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"g": ["A"] * 80 + ["B"] * 20, "x": rng.normal(0, 1, 100)})


_SPEC = SubgroupSpec(name="b-group", conditions={"g": ["B"]})


def test_copied_subgroup_scores_low():
    real = _real()
    # synthetic copies the subgroup rows verbatim; everything else is far away
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] < 0.1
    assert res.per_column["b-group"]["risk_ratio"] < 0.1


def test_uniformly_close_synthetic_scores_high():
    real = _real()
    syn = real.copy()
    syn["x"] = syn["x"] + 0.3  # same offset everywhere -> no concentration
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] > 0.5


def test_worst_subgroup_drives_score():
    real = _real()
    far = real[real.g == "A"].copy()
    far["x"] = far["x"] + 50
    syn = pd.concat([far, real[real.g == "B"].copy()], ignore_index=True)
    safe = SubgroupSpec(name="a-group", conditions={"g": ["A"]})
    res = MinorityPrivacyRisk(subgroup_specs=[safe, _SPEC]).compute(real, syn, _meta())
    assert res.scalars["score"] < 0.1  # min over specs, not mean
    assert res.scalars["mean_risk_ratio"] > res.scalars["worst_risk_ratio"]


def test_tiny_subgroup_skipped():
    spec = SubgroupSpec(name="tiny", conditions={"x": (100.0, None)})
    real = _real()
    res = MinorityPrivacyRisk(subgroup_specs=[spec]).compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert any("tiny" in n for n in res.notes)


def test_no_specs_scores_one_with_note():
    real = _real()
    res = MinorityPrivacyRisk().compute(real, real.copy(), _meta())
    assert res.scalars["score"] == 1.0
    assert res.notes


def test_gower_backend_runs():
    real = _real()
    res = MinorityPrivacyRisk(subgroup_specs=[_SPEC], distance="gower").compute(
        real, real.copy(), _meta()
    )
    assert 0.0 <= res.scalars["score"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/preservation/test_minority_privacy_risk.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/syneva/preservation/minority_privacy_risk.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.distance import gower_matrix
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.preservation._util import _EPS, runnable_specs
from syneva.preservation.spec import SubgroupSpec


@registry.register
@dataclass
class MinorityPrivacyRisk:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="minority_privacy_risk",
        c="preservation",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    subgroup_specs: list[SubgroupSpec] = field(default_factory=list)
    distance: str = "euclidean"
    min_rows: int = 10
    cap: int = 2000
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        specs, notes = runnable_specs(self.subgroup_specs, meta)
        if not specs:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable subgroup specs"],
            )
        rng = np.random.default_rng(self.random_state)

        syn = synthetic
        if len(syn) > self.cap:
            syn = syn.iloc[rng.choice(len(syn), self.cap, replace=False)]
            notes.append(f"synthetic capped at {self.cap} rows")

        member_mask = np.zeros(len(real), dtype=bool)
        for sp in specs:
            member_mask |= sp.matches(real).to_numpy()
        member_idx = np.flatnonzero(member_mask)
        other_idx = np.flatnonzero(~member_mask)
        if len(member_idx) >= self.cap:
            kept_idx = rng.choice(member_idx, self.cap, replace=False)
            notes.append(f"real subgroup rows capped at {self.cap}")
        elif len(real) > self.cap:
            fill = rng.choice(other_idx, self.cap - len(member_idx), replace=False)
            kept_idx = np.concatenate([member_idx, fill])
            notes.append(f"real rows capped at {self.cap} (all subgroup members kept)")
        else:
            kept_idx = np.arange(len(real))
        real_kept = real.iloc[kept_idx].reset_index(drop=True)

        if self.distance == "gower":
            dcr = gower_matrix(real_kept, syn, meta).min(axis=1)
        else:
            x_real, x_syn = encode_pair(real_kept, syn, meta)
            nn = NearestNeighbors(n_neighbors=1).fit(x_syn)
            dcr = nn.kneighbors(x_real)[0][:, 0]

        overall_med = float(np.median(dcr))
        per_column: dict[str, dict[str, float]] = {}
        ratios: list[float] = []
        for sp in specs:
            smask = sp.matches(real_kept).to_numpy()
            if int(smask.sum()) < self.min_rows:
                notes.append(f"subgroup '{sp.name}' skipped (<{self.min_rows} rows)")
                continue
            sub_med = float(np.median(dcr[smask]))
            if overall_med <= _EPS:
                ratio = 1.0
                notes.append(
                    f"subgroup '{sp.name}': overall distances ~0 (global copying); "
                    "no concentration measurable"
                )
            else:
                ratio = sub_med / (overall_med + _EPS)
            ratios.append(ratio)
            per_column[sp.name] = {
                "risk_ratio": ratio,
                "median_dcr_subgroup": sub_med,
                "median_dcr_overall": overall_med,
            }
        if not ratios:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable subgroup specs"],
            )
        scores = [1.0 if r >= 1.0 else float(max(0.0, r)) for r in ratios]
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": min(scores),
                "mean_risk_ratio": sum(ratios) / len(ratios),
                "worst_risk_ratio": min(ratios),
            },
            per_column=per_column,
            notes=notes,
        )
```

Append to `src/syneva/preservation/__init__.py`:

```python
from syneva.preservation import (
    minority_privacy_risk as minority_privacy_risk,  # side-effect: registers metric
)
```

`src/syneva/core/metric_info.py` — `METRIC_INFO`:

```python
    "minority_privacy_risk": "Whether disclosure risk concentrates on declared subgroups rather than spreading evenly across the data.",
```

`RAW_HINT`:

```python
    "minority_privacy_risk": "Ratio of the subgroup's median distance-to-nearest-synthetic-record to the overall median, capped at 1: 1 = no concentrated risk, near 0 = subgroup members are much closer to synthetic records than average. Higher is better.",
```

`METRIC_NAMES`:

```python
    "minority_privacy_risk": "Minority privacy risk",
```

`_SCALAR_LABELS`:

```python
    "mean_risk_ratio": "Mean risk ratio",
    "worst_risk_ratio": "Worst risk ratio",
    "risk_ratio": "Risk ratio",
    "median_dcr_subgroup": "Median distance (subgroup)",
    "median_dcr_overall": "Median distance (overall)",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/preservation/ tests/unit/core/test_metric_info.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/preservation tests/unit/preservation src/syneva/core/metric_info.py
git commit -m "feat(preservation): minority_privacy_risk metric"
```

---

### Task 8: Runner + benchmark plumbing, wiring integration test

**Files:**
- Modify: `src/syneva/core/runner.py` (`_instantiate` line 23; `evaluate` params line 40; `evaluate_with` params line 79; gating after line 146)
- Modify: `src/syneva/benchmark/engine.py` (`benchmark` params line 15, forward line 41)
- Test: `tests/integration/test_preservation_wiring.py`

**Interfaces:**
- Consumes: `SubgroupSpec` (Task 1); metrics from Tasks 2–7; existing `_instantiate`/`evaluate` machinery.
- Produces: `evaluate(..., subgroup_specs: list[SubgroupSpec] | None = None)` and the same on `evaluate_with`/`benchmark`. Drop rule: when `subgroup_specs` is falsy, any selected metric whose constructor has a `subgroup_specs` param is removed from the selection (Task 9's UI relies on this).

- [ ] **Step 1: Write the failing test**

`tests/integration/test_preservation_wiring.py`:

```python
import numpy as np
import pandas as pd

import syneva
from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

_AUTO = {"rare_category_retention", "tail_coverage", "minority_class_density"}
_SUBGROUP = {"subgroup_fidelity", "minority_utility_gap", "minority_privacy_risk"}


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "g": ["A"] * 160 + ["B"] * 40,
            "x": rng.normal(0, 1, 200),
        }
    )


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def test_core_tier_runs_no_preservation():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core",))
    assert not {r.spec.name for r in rep.results} & (_AUTO | _SUBGROUP)


def test_auto_metrics_run_by_default_subgroup_metrics_dropped():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core", "extended"))
    names = {r.spec.name for r in rep.results}
    assert _AUTO <= names
    assert not names & _SUBGROUP


def test_subgroup_metrics_run_with_specs():
    rep = syneva.evaluate(
        _df(),
        _df(),
        _meta(),
        tiers=("core", "extended"),
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    by = {r.spec.name: r for r in rep.results}
    assert _SUBGROUP <= set(by)
    for name in _AUTO | _SUBGROUP:
        assert by[name].error is None, f"{name}: {by[name].error}"
        assert 0.0 <= by[name].scalars["score"] <= 1.0


def test_benchmark_forwards_subgroup_specs():
    res = syneva.benchmark(
        _df(),
        {"cand": _df()},
        _meta(),
        tiers=("core", "extended"),
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    names = {r.spec.name for r in res.reports["cand"].results}
    assert _SUBGROUP <= names
    assert "preservation" in res.reports["cand"].aggregated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_preservation_wiring.py -v`
Expected: FAIL — `evaluate() got an unexpected keyword argument 'subgroup_specs'`

- [ ] **Step 3: Implement**

`src/syneva/core/runner.py`:

1. Add to the `TYPE_CHECKING` block: `from syneva.preservation.spec import SubgroupSpec`.
2. Replace `_instantiate` with:

```python
def _instantiate(
    cls, utility_tasks, fairness_specs, subgroup_specs, distance, holdout, random_state
):
    """Instantiate a metric, passing only the kwargs its constructor accepts."""
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "tasks" in params:
        kwargs["tasks"] = utility_tasks
    if "specs" in params:
        kwargs["specs"] = fairness_specs if fairness_specs is not None else []
    if "subgroup_specs" in params:
        kwargs["subgroup_specs"] = subgroup_specs if subgroup_specs is not None else []
    if "distance" in params:
        kwargs["distance"] = distance
    if "holdout" in params:
        kwargs["holdout"] = holdout
    if "random_state" in params:
        kwargs["random_state"] = random_state
    return cls(**kwargs)
```

3. Add the parameter `subgroup_specs: list[SubgroupSpec] | None = None,` to **both** `evaluate` (after `fairness_specs`/`run_fairness`) and `evaluate_with`, and forward it in `evaluate`'s call to `evaluate_with` (`subgroup_specs=subgroup_specs,`).
4. Add the drop rule directly after the fairness gate (`if not run_fairness: ...`, line 145–146):

```python
    if not subgroup_specs:
        # preservation metrics that need subgroups are dropped, not skipped,
        # so spec-less runs stay noise-free
        selected = [
            c for c in selected if "subgroup_specs" not in inspect.signature(c).parameters
        ]
```

5. Update the `_instantiate` call site (line 158) to
   `_instantiate(cls, utility_tasks, fairness_specs, subgroup_specs, distance, holdout, random_state)`.

`src/syneva/benchmark/engine.py`: add `subgroup_specs=None,` to `benchmark`'s signature (after `fairness_specs=None,`) and forward `subgroup_specs=subgroup_specs,` in the `evaluate(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/integration/ tests/unit -v`
Expected: all PASS (including existing fairness/utility wiring and golden tests — the golden must NOT change)

- [ ] **Step 5: Commit**

```bash
git add src/syneva/core/runner.py src/syneva/benchmark/engine.py tests/integration/test_preservation_wiring.py
git commit -m "feat(preservation): runner subgroup_specs plumbing + benchmark passthrough"
```

---

### Task 9: UI — dimension list, subgroup builder, core passthrough

**Files:**
- Modify: `src/syneva/ui/core.py` (`run_report`, lines 90–143)
- Modify: `src/syneva/ui/app.py` (dimension list line ~88; new sidebar section after "5 · Fairness" line ~123; cfg dict line ~144; `run_report` call line ~295)
- Test: `tests/unit/ui/test_preservation_ui.py`

**Interfaces:**
- Consumes: `SubgroupSpec` (public API), `run_report` (existing), runner drop rule (Task 8).
- Produces: `core.run_report(..., subgroup_specs: list | None = None)` forwarded on both the preset and manual paths.

- [ ] **Step 1: Write the failing test**

`tests/unit/ui/test_preservation_ui.py`:

```python
import numpy as np
import pandas as pd

from syneva import SubgroupSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.ui import core


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"g": ["A"] * 160 + ["B"] * 40, "x": rng.normal(0, 1, 200)})


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC),
        }
    )


def test_run_report_forwards_subgroup_specs():
    rep = core.run_report(
        _df(),
        _df(),
        _meta(),
        selected_names=["subgroup_fidelity", "rare_category_retention"],
        subgroup_specs=[SubgroupSpec(name="b", conditions={"g": ["B"]})],
    )
    names = {r.spec.name for r in rep.results}
    assert {"subgroup_fidelity", "rare_category_retention"} <= names


def test_run_report_without_specs_drops_subgroup_metrics():
    rep = core.run_report(
        _df(),
        _df(),
        _meta(),
        selected_names=["subgroup_fidelity", "rare_category_retention"],
    )
    names = {r.spec.name for r in rep.results}
    assert "rare_category_retention" in names
    assert "subgroup_fidelity" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_preservation_ui.py -v`
Expected: FAIL — `run_report() got an unexpected keyword argument 'subgroup_specs'`

- [ ] **Step 3: Implement**

`src/syneva/ui/core.py`:
- Add `subgroup_specs: list | None = None,` to `run_report`'s signature (after `fairness_specs`).
- Forward `subgroup_specs=subgroup_specs,` in **both** `evaluate_with` calls (preset path and manual path).

`src/syneva/ui/app.py`:
- Line ~88: extend the dimension loop list to
  `["congruence", "coverage", "compliance", "utility", "fairness", "preservation"]`.
- After the "5 · Fairness" section, add a "6 · Preservation" section:

```python
        st.header("6 · Preservation")
        preservation_subgroup_selected = any(
            m["c"] == "preservation" and m["name"] in selected and m["name"] in
            ("subgroup_fidelity", "minority_utility_gap", "minority_privacy_risk")
            for m in catalog
        )
        subgroup_specs = None
        if preservation_subgroup_selected:
            from syneva import SubgroupSpec

            sg_name = st.text_input("Subgroup name", value="subgroup 1")
            sg_cols = st.multiselect("Subgroup condition column(s)", options=list(real.columns))
            conditions: dict = {}
            for col in sg_cols:
                if pd.api.types.is_numeric_dtype(real[col]):
                    lo = st.number_input(f"'{col}' min", value=float(real[col].min()), key=f"sg_lo_{col}")
                    hi = st.number_input(f"'{col}' max", value=float(real[col].max()), key=f"sg_hi_{col}")
                    conditions[col] = (lo, hi)
                else:
                    vals = st.multiselect(
                        f"'{col}' values", options=sorted(real[col].dropna().unique().tolist()),
                        key=f"sg_vals_{col}",
                    )
                    if vals:
                        conditions[col] = vals
            if conditions:
                subgroup_specs = [SubgroupSpec(name=sg_name, conditions=conditions)]
            else:
                st.info(
                    "No subgroup defined — only the automatic preservation metrics will run."
                )
        else:
            st.caption("Select a subgroup metric to configure subgroups.")
```

- Add `"subgroup_specs": subgroup_specs,` to the returned cfg dict (with the other keys around line 144; initialize `subgroup_specs = None` before the section so the no-preservation path defines it).
- In the run block (~line 295), pass `subgroup_specs=cfg["subgroup_specs"],` to `core.run_report`.
- Do NOT add a blocking warning for a missing subgroup (unlike fairness) — the info note above is the designed behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/ui -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/syneva/ui/core.py src/syneva/ui/app.py tests/unit/ui/test_preservation_ui.py
git commit -m "feat(ui): preservation dimension + subgroup builder"
```

---

### Task 10: Full verification + roadmap update

**Files:**
- Modify: `docs/superpowers/specs/2026-06-23-syneva-master-roadmap.md` (C4 row line 128, C5 row line 129, open question line 187)

**Interfaces:**
- Consumes: everything above.
- Produces: green suite, coverage ≥ 85 %, updated roadmap.

- [ ] **Step 1: Run the full test suite with coverage**

Run: `uv run pytest`
Expected: all tests PASS, coverage ≥ 85 % (the gate is in pyproject). If any property-based or guard test trips on the new metrics, fix the metric — do not weaken the test.

- [ ] **Step 2: Confirm the golden snapshot is untouched**

Run: `git status --short tests/golden/`
Expected: no output (no golden file modified).

- [ ] **Step 3: Update the roadmap**

In `docs/superpowers/specs/2026-06-23-syneva-master-roadmap.md`:
- C4 row: change status `⏭️ next` → `✅ done` (it was completed earlier but never marked).
- C5 row: append status `✅ done`.
- Open-questions line "C5: is minority/outlier preservation its own `preservation` dimension or an extension of Coverage?" → append " **Resolved: own dimension** (see 2026-08-18-syneva-preservation-5.md)."

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-23-syneva-master-roadmap.md
git commit -m "docs: mark C4/C5 done in master roadmap"
```

- [ ] **Step 5: Hand off**

Invoke superpowers:finishing-a-development-branch for the merge decision (user picks; historically "1" = merge to main).
