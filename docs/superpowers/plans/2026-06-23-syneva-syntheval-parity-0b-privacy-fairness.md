# SynthEval Parity 0b — Privacy + Fairness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three privacy metrics + a new Fairness dimension (statistical parity) so syneva covers all 21 SynthEval metrics, wiring a `FairnessSpec` config through `evaluate()` exactly like the existing `utility_tasks` pattern.

**Architecture:** `hitting_rate`/`epsilon_identifiability`/`attribute_disclosure` are self-contained `@registry.register` Compliance metrics (the latter two reuse `encode_pair`; attribute disclosure reuses the `Metadata.sensitive` flag). Fairness is a new `spec.c` value with a `FairnessSpec` dataclass and a `statistical_parity` metric, threaded via new `fairness_specs`/`run_fairness` args on `evaluate()`/`evaluate_with()` (the runner's `_instantiate` is generalized to pass `specs=`). The parity manifest then proves 21/21 coverage.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-23-syneva-syntheval-parity-0b-privacy-fairness.md`.

---

## CRITICAL environment notes (read before every task)

- Use `uv`. Targeted test runs MUST use `--no-cov` (the repo's `addopts` has `--cov-fail-under=85`, which fails on subset runs): e.g. `uv run pytest tests/unit/compliance/test_hitting_rate.py --no-cov -v`. Only the full `uv run pytest` enforces the gate.
- After registering a metric you MUST add its `src/syneva/core/metric_info.py` entries (METRIC_NAMES, METRIC_INFO neutral with no "higher/lower is better", RAW_HINT with correct direction, _SCALAR_LABELS for each new scalar key) in the SAME task — the full-registry guard tests (`tests/unit/core/test_metric_info.py`) fail otherwise. Re-run that guard file each task.
- Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, then commit. If a commit aborts because a hook modified files, `git add -A` and re-run.
- Scores are floats in `[0, 1]`, 1 = ideal; clamp with `float(min(1.0, max(0.0, x)))`. `_EPS = 1e-12`.

## File structure

- Create: `src/syneva/compliance/hitting_rate.py`
- Create: `src/syneva/compliance/extended/epsilon_identifiability.py`, `attribute_disclosure.py`
- Create: `src/syneva/fairness/__init__.py`, `src/syneva/fairness/spec.py`, `src/syneva/fairness/statistical_parity.py`
- Modify: `src/syneva/compliance/__init__.py`, `src/syneva/compliance/extended/__init__.py`
- Modify: `src/syneva/core/metric.py` (add `"fairness"` to the `C` Literal)
- Modify: `src/syneva/core/metric_info.py` (C_INFO + entries for the 4 metrics)
- Modify: `src/syneva/core/runner.py` (fairness args + `_instantiate`)
- Modify: `src/syneva/__init__.py` (register `fairness` package + export `FairnessSpec`)
- Modify: `src/syneva/ui/core.py`, `src/syneva/ui/app.py` (fairness config)
- Modify: `tests/parity/test_syntheval_parity.py` (flip 4 entries)
- Test files under `tests/unit/compliance/`, `tests/unit/compliance/extended/`, `tests/unit/fairness/`, `tests/integration/`

---

## Task 1: `hitting_rate` (Compliance, core) + regenerate golden

**Files:**
- Create: `src/syneva/compliance/hitting_rate.py`
- Modify: `src/syneva/compliance/__init__.py`, `src/syneva/core/metric_info.py`
- Test: `tests/unit/compliance/test_hitting_rate.py`
- Regenerate: `tests/golden/adult_income_good.scorecard.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/compliance/test_hitting_rate.py
import pandas as pd

from syneva.compliance.hitting_rate import HittingRate
from syneva.core.metadata import Metadata


def test_identical_synthetic_full_hit():
    df = pd.DataFrame({"x": list(range(100))})
    r = HittingRate().compute(df, df, Metadata.infer(df))
    assert r.scalars["hit_rate"] == 1.0
    assert r.scalars["score"] == 0.0


def test_disjoint_no_hits():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(1000, 1100))})
    r = HittingRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["hit_rate"] == 0.0
    assert r.scalars["score"] == 1.0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/test_hitting_rate.py --no-cov -v` (module not found).

- [ ] **Step 3: Write the metric**

```python
# src/syneva/compliance/hitting_rate.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_CAP = 2000


@registry.register
class HittingRate:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="hitting_rate",
        c="compliance",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        num = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.NUMERIC]
        cat = [n for n, cm in meta.columns.items() if cm.dtype is ColumnType.CATEGORICAL]
        if not num and not cat:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "hit_rate": 0.0},
                notes=["no usable columns"],
            )
        rng = np.random.default_rng(42)
        notes: list[str] = []
        r, s = real, synthetic
        if len(r) > _CAP:
            r = r.iloc[rng.choice(len(r), _CAP, replace=False)]
            notes.append("real capped at 2000 rows")
        if len(s) > _CAP:
            s = s.iloc[rng.choice(len(s), _CAP, replace=False)]
            notes.append("synthetic capped at 2000 rows")
        thresh = {
            n: float(pd.to_numeric(real[n], errors="coerce").max() - pd.to_numeric(real[n], errors="coerce").min()) / 30.0
            for n in num
        }
        r_num = {n: pd.to_numeric(r[n], errors="coerce").to_numpy() for n in num}
        s_num = {n: pd.to_numeric(s[n], errors="coerce").to_numpy() for n in num}
        r_cat = {n: r[n].astype("string").fillna("__NA__").to_numpy() for n in cat}
        s_cat = {n: s[n].astype("string").fillna("__NA__").to_numpy() for n in cat}
        hits = 0
        for i in range(len(r)):
            mask = np.ones(len(s), dtype=bool)
            for n in num:
                if thresh[n] <= 0:
                    mask &= s_num[n] == r_num[n][i]
                else:
                    mask &= np.abs(s_num[n] - r_num[n][i]) <= thresh[n]
                if not mask.any():
                    break
            if mask.any():
                for n in cat:
                    mask &= s_cat[n] == r_cat[n][i]
                    if not mask.any():
                        break
            if mask.any():
                hits += 1
        hit_rate = float(hits / len(r)) if len(r) else 0.0
        score = float(min(1.0, max(0.0, 1.0 - hit_rate)))
        return MetricResult(spec=self.spec, scalars={"score": score, "hit_rate": hit_rate}, notes=notes)
```

- [ ] **Step 4: Register** — in `src/syneva/compliance/__init__.py` add:

```python
from syneva.compliance import hitting_rate as hitting_rate  # side-effect: registers HittingRate
```

- [ ] **Step 5: metric-info entries** — add:
- `METRIC_NAMES`: `"hitting_rate": "Hitting rate",`
- `METRIC_INFO`: `"hitting_rate": "How often a real record has a near-identical synthetic counterpart.",`
- `RAW_HINT`: `"hitting_rate": "Fraction of real records reproduced by the synthetic data, 0 to 1. Lower is better.",`
- `_SCALAR_LABELS`: `"hit_rate": "Hit rate",`

- [ ] **Step 6: Run tests** — `uv run pytest tests/unit/compliance/test_hitting_rate.py tests/unit/core/test_metric_info.py --no-cov -v` (all pass).

- [ ] **Step 7: Regenerate the golden snapshot** — `hitting_rate` is core, so the `tiers=("core",)` golden gains it (additive, intentional):

Run: `uv run pytest tests/integration/test_golden.py --update-golden --no-cov -q` then confirm `uv run pytest tests/integration/test_golden.py --no-cov -q` passes.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(compliance): hitting rate (SynthEval parity); regen golden"
```

---

## Task 2: `epsilon_identifiability` (Compliance, extended)

**Files:**
- Create: `src/syneva/compliance/extended/epsilon_identifiability.py`
- Modify: `src/syneva/compliance/extended/__init__.py`, `src/syneva/core/metric_info.py`
- Test: `tests/unit/compliance/extended/test_epsilon_identifiability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/compliance/extended/test_epsilon_identifiability.py
import numpy as np
import pandas as pd

from syneva.compliance.extended.epsilon_identifiability import EpsilonIdentifiability
from syneva.core.metadata import Metadata


def test_synthetic_equals_real_high_risk():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    r = EpsilonIdentifiability().compute(df, df, Metadata.infer(df))
    assert r.scalars["identifiability_risk"] > 0.9
    assert r.scalars["score"] < 0.1


def test_far_synthetic_low_risk():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    syn = pd.DataFrame({"x": rng.normal(loc=50, size=200), "y": rng.normal(loc=50, size=200)})
    r = EpsilonIdentifiability().compute(real, syn, Metadata.infer(real))
    assert r.scalars["identifiability_risk"] < 0.1
    assert r.scalars["score"] > 0.9
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/extended/test_epsilon_identifiability.py --no-cov -v`.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/compliance/extended/epsilon_identifiability.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class EpsilonIdentifiability:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="epsilon_identifiability",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        x_real, x_syn = encode_pair(real, synthetic, meta)
        if x_real.shape[1] == 0 or len(x_real) < 2 or len(x_syn) < 1:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "identifiability_risk": 0.0},
                notes=["insufficient data"],
            )
        d_self = NearestNeighbors(n_neighbors=2).fit(x_real).kneighbors(x_real)[0][:, 1]
        d_syn = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real)[0][:, 0]
        risk = float(np.mean(d_syn < d_self))
        score = float(min(1.0, max(0.0, 1.0 - risk)))
        return MetricResult(
            spec=self.spec,
            scalars={"score": score, "identifiability_risk": risk},
            notes=["unweighted distance; entropy-weighting is a planned refinement"],
        )
```

- [ ] **Step 4: Register** — in `src/syneva/compliance/extended/__init__.py` add:

```python
from syneva.compliance.extended import (
    epsilon_identifiability as epsilon_identifiability,  # side-effect: registers it
)
```

- [ ] **Step 5: metric-info entries**
- `METRIC_NAMES`: `"epsilon_identifiability": "Epsilon identifiability risk",`
- `METRIC_INFO`: `"epsilon_identifiability": "Whether real records are closer to a synthetic record than to their own nearest real neighbour.",`
- `RAW_HINT`: `"epsilon_identifiability": "Fraction of real records that are identifiable, 0 to 1. Lower is better.",`
- `_SCALAR_LABELS`: `"identifiability_risk": "Identifiability risk",`

- [ ] **Step 6: Run tests** — `uv run pytest tests/unit/compliance/extended/test_epsilon_identifiability.py tests/unit/core/test_metric_info.py --no-cov -v`.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(compliance/extended): epsilon identifiability risk (SynthEval parity)"
```

---

## Task 3: `attribute_disclosure` (Compliance, extended)

**Files:**
- Create: `src/syneva/compliance/extended/attribute_disclosure.py`
- Modify: `src/syneva/compliance/extended/__init__.py`, `src/syneva/core/metric_info.py`
- Test: `tests/unit/compliance/extended/test_attribute_disclosure.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/compliance/extended/test_attribute_disclosure.py
import pandas as pd

from syneva.compliance.extended.attribute_disclosure import AttributeDisclosure
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta_sens():
    return Metadata(
        columns={
            "qi": ColumnMetadata(name="qi", dtype=ColumnType.NUMERIC),
            "secret": ColumnMetadata(name="secret", dtype=ColumnType.CATEGORICAL, sensitive=True),
        }
    )


def test_copied_synthetic_high_disclosure():
    qi = list(range(100))
    secret = ["A" if v < 50 else "B" for v in qi]
    df = pd.DataFrame({"qi": qi, "secret": secret})
    r = AttributeDisclosure().compute(df, df, _meta_sens())
    assert r.scalars["disclosure_rate"] > 0.9
    assert r.scalars["score"] < 0.1


def test_no_sensitive_columns_skips():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    r = AttributeDisclosure().compute(df, df, Metadata.infer(df))
    assert r.scalars["score"] == 1.0
    assert any("sensitive" in n for n in r.notes)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/compliance/extended/test_attribute_disclosure.py --no-cov -v`.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/compliance/extended/attribute_disclosure.py
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from syneva.compliance._encode import encode_pair
from syneva.core.metadata import ColumnType, Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry

_USABLE = (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.BOOLEAN)


@registry.register
class AttributeDisclosure:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="attribute_disclosure",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None
        sensitive = [n for n, cm in meta.columns.items() if cm.sensitive]
        qi = [
            n
            for n, cm in meta.columns.items()
            if not cm.sensitive and cm.dtype in _USABLE
        ]
        if not sensitive or not qi:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no sensitive columns or no quasi-identifiers; skipped"],
            )
        qi_meta = Metadata(columns={n: meta.columns[n] for n in qi})
        x_real, x_syn = encode_pair(real, synthetic, qi_meta)
        if x_real.shape[1] == 0 or len(x_syn) < 1:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0, "disclosure_rate": 0.0},
                notes=["no encodable quasi-identifiers; skipped"],
            )
        idx = NearestNeighbors(n_neighbors=1).fit(x_syn).kneighbors(x_real, return_distance=False)[:, 0]
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
        )
```

- [ ] **Step 4: Register** — in `src/syneva/compliance/extended/__init__.py` add:

```python
from syneva.compliance.extended import (
    attribute_disclosure as attribute_disclosure,  # side-effect: registers it
)
```

- [ ] **Step 5: metric-info entries**
- `METRIC_NAMES`: `"attribute_disclosure": "Attribute disclosure risk",`
- `METRIC_INFO`: `"attribute_disclosure": "Whether a sensitive attribute can be inferred from the quasi-identifiers via the nearest synthetic record.",`
- `RAW_HINT`: `"attribute_disclosure": "Fraction of records whose sensitive attribute is correctly inferred, 0 to 1. Lower is better.",`
- `_SCALAR_LABELS`: `"disclosure_rate": "Disclosure rate",`

- [ ] **Step 6: Run tests** — `uv run pytest tests/unit/compliance/extended/test_attribute_disclosure.py tests/unit/core/test_metric_info.py --no-cov -v`.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(compliance/extended): attribute disclosure risk (SynthEval parity)"
```

---

## Task 4: Fairness dimension scaffolding (`C` Literal, `FairnessSpec`, package, export)

**Files:**
- Modify: `src/syneva/core/metric.py`
- Modify: `src/syneva/core/metric_info.py`
- Create: `src/syneva/fairness/__init__.py`, `src/syneva/fairness/spec.py`
- Modify: `src/syneva/__init__.py`
- Test: `tests/unit/fairness/test_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/fairness/test_spec.py
from syneva import FairnessSpec
from syneva.core.metric_info import describe_c


def test_fairness_spec_fields():
    sp = FairnessSpec(protected_attribute="sex", outcome="y")
    assert sp.protected_attribute == "sex"
    assert sp.outcome == "y"
    assert sp.favorable_outcome is None


def test_fairness_dimension_described():
    assert describe_c("fairness") != ""
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/fairness/test_spec.py --no-cov -v` (ImportError / empty describe).

- [ ] **Step 3: Add `"fairness"` to the `C` Literal** in `src/syneva/core/metric.py`. Change the `C = Literal[...]` block to include `"fairness"`:

```python
C = Literal[
    "congruence",
    "coverage",
    "compliance",
    "utility",
    "fairness",
    "constraint",
    "completeness",
    "comprehension",
    "consistency",
]
```

- [ ] **Step 4: Add `C_INFO["fairness"]`** in `src/syneva/core/metric_info.py`:

```python
    "fairness": "Does the synthetic data preserve the real data's fairness across protected groups?",
```

- [ ] **Step 5: Create the package + spec**

```python
# src/syneva/fairness/__init__.py
from syneva.fairness import (
    statistical_parity as statistical_parity,  # side-effect: registers StatisticalParity
)
```

```python
# src/syneva/fairness/spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FairnessSpec:
    """Declares a fairness comparison: a protected attribute, an outcome column,
    and (optionally) which outcome value is the favorable one (defaults to the
    maximum outcome value, e.g. 1 / True)."""

    protected_attribute: str
    outcome: str
    favorable_outcome: Any = None
```

Note: `fairness/__init__.py` imports `statistical_parity`, which does not exist until Task 5. To keep this task's tests green, create a minimal placeholder now and replace it in Task 5 — OR (preferred) reorder so the `__init__` import is added in Task 5. **Do this:** in Task 4 leave `src/syneva/fairness/__init__.py` EMPTY (just create the file); add the `statistical_parity` side-effect import to it in Task 5. This keeps Task 4 importable.

- [ ] **Step 6: Export `FairnessSpec`** — in `src/syneva/__init__.py` add `from syneva.fairness.spec import FairnessSpec` and add `"FairnessSpec"` to `__all__`.

- [ ] **Step 7: Run tests** — `uv run pytest tests/unit/fairness/test_spec.py tests/unit/core/test_metric_info.py --no-cov -v` (create `tests/unit/fairness/__init__.py` empty if needed).

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(fairness): add fairness dimension + FairnessSpec config"
```

---

## Task 5: `statistical_parity` (Fairness, extended)

**Files:**
- Create: `src/syneva/fairness/statistical_parity.py`
- Modify: `src/syneva/fairness/__init__.py`, `src/syneva/__init__.py` (register fairness package), `src/syneva/core/metric_info.py`
- Test: `tests/unit/fairness/test_statistical_parity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/fairness/test_statistical_parity.py
import pandas as pd

from syneva import FairnessSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.fairness.statistical_parity import StatisticalParity


def _meta_gy():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def _biased(seed_a_pos, seed_b_pos):
    # group A: seed_a_pos ones then rest zeros (50 each); group B similar
    y = [1] * seed_a_pos + [0] * (50 - seed_a_pos) + [1] * seed_b_pos + [0] * (50 - seed_b_pos)
    return pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})


def test_preserved_bias_low_drift():
    real = _biased(40, 10)  # SPD = 0.8 - 0.2 = 0.6
    syn = _biased(40, 10)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] < 0.05
    assert r.scalars["score"] > 0.95


def test_destroyed_bias_high_drift():
    real = _biased(40, 10)  # SPD 0.6
    syn = _biased(25, 25)  # SPD 0.0
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] > 0.5
    assert r.scalars["score"] < 0.5


def test_no_specs_score_one():
    df = _biased(40, 10)
    r = StatisticalParity(specs=[]).compute(df, df, _meta_gy())
    assert r.scalars["score"] == 1.0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/fairness/test_statistical_parity.py --no-cov -v`.

- [ ] **Step 3: Write the metric**

```python
# src/syneva/fairness/statistical_parity.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.fairness.spec import FairnessSpec


def _spd(df: pd.DataFrame, protected: str, outcome: str, favorable: object) -> float:
    sub = df[[protected, outcome]].dropna()
    rates = []
    for g in sub[protected].unique():
        gy = sub[sub[protected] == g][outcome]
        if len(gy) == 0:
            continue
        rates.append(float((gy == favorable).mean()))
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


@registry.register
@dataclass
class StatisticalParity:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="statistical_parity",
        c="fairness",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    specs: list[FairnessSpec] = field(default_factory=list)
    random_state: int = 42

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        per_column: dict[str, dict[str, float]] = {}
        notes: list[str] = []
        drifts: list[float] = []
        headline: tuple[float, float, float] | None = None
        for sp in self.specs:
            if sp.protected_attribute not in real.columns or sp.outcome not in real.columns:
                notes.append(
                    f"spec {sp.protected_attribute}->{sp.outcome} skipped (missing column)"
                )
                continue
            favorable = sp.favorable_outcome
            if favorable is None:
                favorable = max(
                    pd.concat([real[sp.outcome], synthetic[sp.outcome]]).dropna().unique()
                )
            spd_r = _spd(real, sp.protected_attribute, sp.outcome, favorable)
            spd_s = _spd(synthetic, sp.protected_attribute, sp.outcome, favorable)
            drift = abs(spd_s - spd_r)
            per_column[f"{sp.protected_attribute}|{sp.outcome}"] = {
                "spd_synthetic": spd_s,
                "spd_real": spd_r,
                "parity_drift": drift,
            }
            drifts.append(drift)
            if headline is None:
                headline = (spd_s, spd_r, drift)
        if not drifts:
            return MetricResult(
                spec=self.spec,
                scalars={"score": 1.0},
                notes=[*notes, "no runnable fairness specs"],
            )
        mean_drift = sum(drifts) / len(drifts)
        score = float(min(1.0, max(0.0, 1.0 - mean_drift)))
        spd_s, spd_r, drift = headline  # type: ignore[misc]
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "spd_synthetic": spd_s,
                "spd_real": spd_r,
                "parity_drift": drift,
            },
            per_column=per_column,
            notes=notes,
        )
```

- [ ] **Step 4: Register** — set `src/syneva/fairness/__init__.py` to:

```python
from syneva.fairness import (
    statistical_parity as statistical_parity,  # side-effect: registers StatisticalParity
)
```

And in `src/syneva/__init__.py`, alongside the other built-in package imports (`from syneva import compliance ...`), add `from syneva import fairness as _fairness  # noqa: F401` so the metric registers on `import syneva`.

- [ ] **Step 5: metric-info entries**
- `METRIC_NAMES`: `"statistical_parity": "Statistical parity difference",`
- `METRIC_INFO`: `"statistical_parity": "Whether the synthetic data preserves the real data's outcome-rate gap between protected groups.",`
- `RAW_HINT`: `"statistical_parity": "Drift = |synthetic parity gap - real parity gap|, 0 to 1. 0 means the real fairness structure is preserved. Lower is better.",`
- `_SCALAR_LABELS`: `"spd_synthetic": "Parity gap (synthetic)",` · `"spd_real": "Parity gap (real)",` · `"parity_drift": "Parity drift",`

- [ ] **Step 6: Run tests** — `uv run pytest tests/unit/fairness/ tests/unit/core/test_metric_info.py --no-cov -v`.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(fairness): statistical parity difference (absolute + drift)"
```

---

## Task 6: Runner plumbing — `fairness_specs` / `run_fairness`

**Files:**
- Modify: `src/syneva/core/runner.py`
- Test: `tests/integration/test_fairness_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_fairness_wiring.py
import pandas as pd

import syneva
from syneva import FairnessSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _df():
    y = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    return pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})


def _meta():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_fairness_off_by_default():
    rep = syneva.evaluate(_df(), _df(), _meta(), tiers=("core", "extended"))
    assert "statistical_parity" not in {r.spec.name for r in rep.results}


def test_fairness_runs_when_enabled():
    rep = syneva.evaluate(
        _df(), _df(), _meta(),
        tiers=("core", "extended"),
        run_fairness=True,
        fairness_specs=[FairnessSpec("g", "y")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "statistical_parity" in by
    assert by["statistical_parity"].error is None
    assert by["statistical_parity"].scalars["parity_drift"] < 0.05
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/integration/test_fairness_wiring.py --no-cov -v` (TypeError: unexpected `run_fairness`).

- [ ] **Step 3: Generalize `_instantiate`** in `src/syneva/core/runner.py`. Replace it with:

```python
def _instantiate(cls, utility_tasks, fairness_specs, random_state):
    """Instantiate a metric, passing only the kwargs its constructor accepts."""
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "tasks" in params:
        kwargs["tasks"] = utility_tasks
    if "specs" in params:
        kwargs["specs"] = fairness_specs if fairness_specs is not None else []
    if "random_state" in params:
        kwargs["random_state"] = random_state
    return cls(**kwargs)
```

- [ ] **Step 4: Thread the new args through `evaluate` and `evaluate_with`**

Add a TYPE_CHECKING import in `src/syneva/core/runner.py`:

```python
if TYPE_CHECKING:
    from syneva.fairness.spec import FairnessSpec
    from syneva.utility.task import UtilityTask
```

In `evaluate(...)`, add params after `run_utility`:

```python
    fairness_specs: list[FairnessSpec] | None = None,
    run_fairness: bool = False,
```

and forward them in the `evaluate_with(...)` call (add `fairness_specs=fairness_specs, run_fairness=run_fairness,`).

In `evaluate_with(...)`, add the same two params; after the existing utility filter block add:

```python
    if not run_fairness:
        selected = [c for c in selected if c.spec.c != "fairness"]
```

and update the instantiation call site (currently `inst = _instantiate(cls, utility_tasks, random_state)`) to:

```python
            inst = _instantiate(cls, utility_tasks, fairness_specs, random_state)
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/integration/test_fairness_wiring.py --no-cov -v` (2 passed).

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(runner): fairness_specs + run_fairness wiring through evaluate()"
```

---

## Task 7: UI wiring — fairness config in the Streamlit app

**Files:**
- Modify: `src/syneva/ui/core.py`, `src/syneva/ui/app.py`
- Test: `tests/unit/ui/test_core.py` (extend)

- [ ] **Step 1: Write the failing test** — append to `tests/unit/ui/test_core.py`:

```python
def test_run_report_runs_fairness_when_selected():
    import pandas as pd

    from syneva import FairnessSpec
    from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

    y = [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40
    df = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})
    meta = Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )
    rep = core.run_report(
        df, df, meta, ["statistical_parity"], fairness_specs=[FairnessSpec("g", "y")]
    )
    by = {r.spec.name: r for r in rep.results}
    assert "statistical_parity" in by
    assert by["statistical_parity"].error is None
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/ui/test_core.py -k fairness --no-cov -v`.

- [ ] **Step 3: Extend `core.run_report`** in `src/syneva/ui/core.py`. Add a `_FAIRNESS_C = "fairness"` constant near `_UTILITY_C`. Change the signature to add `fairness_specs: list | None = None` after `utility_tasks`, and the body to infer + pass run_fairness:

```python
    run_utility = any(cls.spec.c == _UTILITY_C for cls in reg.metrics())
    run_fairness = any(cls.spec.c == _FAIRNESS_C for cls in reg.metrics())
    return evaluate_with(
        reg,
        real=real,
        synthetic=synthetic,
        metadata=metadata,
        tiers=("core", "extended", "custom"),
        utility_tasks=utility_tasks,
        run_utility=run_utility,
        fairness_specs=fairness_specs,
        run_fairness=run_fairness,
        random_state=random_state,
    )
```

- [ ] **Step 4: Run the unit test** — `uv run pytest tests/unit/ui/test_core.py -k fairness --no-cov -v` (passes).

- [ ] **Step 5: Add the fairness sidebar section** in `src/syneva/ui/app.py` `_sidebar()`. After the "Utility tasks" section and before the run button, add (mirroring the utility-tasks block):

```python
        st.header("5 · Fairness")
        fairness_specs = None
        if any(m["c"] == "fairness" for m in catalog if m["name"] in selected):
            from syneva import FairnessSpec

            protected = st.selectbox("Protected attribute", options=list(real.columns))
            outcome = st.selectbox("Outcome column", options=list(real.columns))
            if protected and outcome:
                fairness_specs = [FairnessSpec(protected_attribute=protected, outcome=outcome)]
        else:
            st.caption("Select the statistical-parity metric to configure fairness.")
```

Add `"fairness_specs": fairness_specs,` to the dict `_sidebar()` returns, and in `main()` pass it through to `core.run_report(...)`:

```python
                st.session_state["report"] = core.run_report(
                    cfg["real"],
                    cfg["synthetic"],
                    meta,
                    cfg["selected"],
                    cfg["utility_tasks"],
                    fairness_specs=cfg["fairness_specs"],
                )
```

Also extend the "selected but unconfigured" guard in `main()`: if a fairness metric is selected but `cfg["fairness_specs"]` is None, `st.warning("A fairness metric is selected but no protected attribute/outcome was chosen.")` and skip the run (mirror the utility guard). To support that, also return a `"fairness_selected"` boolean from `_sidebar()` (`any(m["c"]=="fairness" ... in selected)`).

- [ ] **Step 6: Verify import-safety + manual boot** — `uv run pytest tests/unit/ui/test_app_importable.py --no-cov -v`; then `uv run streamlit run src/syneva/ui/app.py --server.headless true --server.port 8602 >/tmp/boot.log 2>&1 &`, after ~6s `curl -s -o /dev/null -w "%{http_code}" http://localhost:8602/` should be 200, no tracebacks in `/tmp/boot.log`, then kill it.

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "feat(ui): fairness config (protected attribute + outcome) in the app"
```

---

## Task 8: Parity manifest → 21/21 + integration + full suite

**Files:**
- Modify: `tests/parity/test_syntheval_parity.py`
- Modify: `tests/integration/test_full_core_scorecard.py`

- [ ] **Step 1: Flip the four manifest entries** in `tests/parity/test_syntheval_parity.py`:

```python
    "hit_rate": ("implemented", "hitting_rate"),
    "eps_risk": ("implemented", "epsilon_identifiability"),
    "att_discl": ("implemented", "attribute_disclosure"),
    "statistical_parity": ("implemented", "statistical_parity"),
```

- [ ] **Step 2: Run the parity test** — `uv run pytest tests/parity/ --no-cov -v` (2 passed; all 21 now `implemented`).

- [ ] **Step 3: Add an integration assertion** — append to `tests/integration/test_full_core_scorecard.py`:

```python
def test_privacy_parity_metrics_present(real_df, syn_good_df, metadata):
    import syneva

    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core", "extended"))
    names = {r.spec.name for r in rep.results}
    for m in ["hitting_rate", "epsilon_identifiability", "attribute_disclosure"]:
        assert m in names, f"{m} missing"
        assert next(r for r in rep.results if r.spec.name == m).error is None
```

- [ ] **Step 4: Run the integration test** — `uv run pytest tests/integration/test_full_core_scorecard.py --no-cov -v`.

- [ ] **Step 5: Full suite + gate** — `uv run pytest -q` (all pass, coverage >= 85%).

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix . && uv run ruff format .
git add -A
git commit -m "test: SynthEval parity 21/21 + privacy/fairness integration"
```

---

## Self-review notes

- **Spec coverage:** hitting_rate (T1) · epsilon_identifiability (T2) · attribute_disclosure (T3) · fairness dimension + FairnessSpec (T4) · statistical_parity absolute+drift (T5) · runner fairness_specs/run_fairness + _instantiate specs (T6) · UI config (T7) · manifest 21/21 + integration (T8). Golden regen handled in T1 (the only new core metric). metric-info entries added per metric (guard-enforced).
- **Type/signature consistency:** all `compute(self, real, synthetic, meta)`; `_instantiate(cls, utility_tasks, fairness_specs, random_state)` matches its call site; `FairnessSpec(protected_attribute, outcome, favorable_outcome=None)` consistent across metric, runner, UI, tests; new scalar keys (`hit_rate`, `identifiability_risk`, `disclosure_rate`, `spd_synthetic`, `spd_real`, `parity_drift`) match their `_SCALAR_LABELS`.
- **No placeholders:** every code step is complete. Task 4 explicitly notes leaving `fairness/__init__.py` empty until Task 5 wires the import, to avoid an import error mid-plan.
