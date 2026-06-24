# Benchmark Engine Core (C4-a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `benchmark(real, candidates)` engine that runs N synthetic candidates through `evaluate()` with one shared config and returns a serializable, rankable `BenchmarkResult`.

**Architecture:** New `src/syneva/benchmark/` package: `normalize.py` (pure cross-candidate normalization), `engine.py` (`benchmark()` + `BenchmarkResult`). Displayed per-C/overall scores are absolute [0,1]; `ranking(normalization=...)` re-normalizes per metric across the cohort to set order only. Reuses `evaluate`, `Report.aggregated`, `_normalize_scalars`.

**Tech Stack:** Python 3.10+, pandas, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-24-syneva-benchmark-4a-engine.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- This is purely ADDITIVE (new package + one export line). No existing metric/runner behavior changes; the golden snapshot must stay green. If an existing test or the golden changes, STOP and report.
- Reuse, don't reimplement: `_normalize_scalars` lives in `src/syneva/core/report.py`; `Report.aggregated` gives per-C means; `_UNSET` is in `src/syneva/core/presets.py`.

## File structure
- Create: `src/syneva/benchmark/__init__.py`, `src/syneva/benchmark/normalize.py`, `src/syneva/benchmark/engine.py`
- Modify: `src/syneva/__init__.py` (export `benchmark`, `BenchmarkResult`)
- Tests: `tests/unit/benchmark/test_normalize.py`, `tests/unit/benchmark/test_engine.py`, `tests/integration/test_benchmark.py`

---

## Task 1: cross-candidate normalization (`normalize.py`)

**Files:** Create `src/syneva/benchmark/normalize.py`, `src/syneva/benchmark/__init__.py` (empty placeholder ok); Test `tests/unit/benchmark/test_normalize.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/benchmark/test_normalize.py
import math

import pytest

from syneva.benchmark.normalize import normalize_across
from syneva.core.errors import SynevaError


def test_absolute_is_identity():
    assert normalize_across([0.2, 0.8, 0.5], "absolute") == [0.2, 0.8, 0.5]


def test_linear_min_max():
    assert normalize_across([0.2, 0.8, 0.5], "linear") == [0.0, 1.0, 0.5]


def test_linear_constant_all_ones():
    assert normalize_across([0.4, 0.4, 0.4], "linear") == [1.0, 1.0, 1.0]


def test_quantile_ranks():
    assert normalize_across([0.2, 0.8, 0.5], "quantile") == [0.0, 1.0, 0.5]


def test_quantile_handles_ties():
    # two tied lowest -> average rank 0.5 each, top -> rank 2
    out = normalize_across([0.3, 0.3, 0.9], "quantile")
    assert out == [0.25, 0.25, 1.0]


def test_quantile_single_is_half():
    assert normalize_across([0.7], "quantile") == [0.5]


def test_normal_zscores_sum_to_zero():
    out = normalize_across([0.2, 0.8, 0.5], "normal")
    assert math.isclose(sum(out), 0.0, abs_tol=1e-9)
    assert out[1] > out[2] > out[0]


def test_normal_constant_all_zero():
    assert normalize_across([0.4, 0.4, 0.4], "normal") == [0.0, 0.0, 0.0]


def test_unknown_mode_raises():
    with pytest.raises(SynevaError, match="normalization"):
        normalize_across([0.1, 0.2], "bogus")
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/benchmark/test_normalize.py --no-cov -v` (ModuleNotFoundError).

- [ ] **Step 3: Implement** — create `src/syneva/benchmark/__init__.py` (empty for now; populated in Task 3) and `src/syneva/benchmark/normalize.py`:

```python
from __future__ import annotations

from syneva.core.errors import SynevaError

_MODES = ("absolute", "linear", "normal", "quantile")


def _average_ranks(values: list[float]) -> list[float]:
    """0-based ranks, ties averaged (e.g. two tied lowest -> 0.5, 0.5)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = sum(range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def normalize_across(values: list[float], mode: str) -> list[float]:
    """Transform one metric's absolute scores across the candidate cohort.

    Used for ranking order ONLY; displayed scores stay absolute. `values` are in
    candidate order; the return list is in the same order.
    """
    n = len(values)
    if mode == "absolute":
        return list(values)
    if mode == "linear":
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0] * n
        return [(v - lo) / (hi - lo) for v in values]
    if mode == "quantile":
        if n < 2:
            return [0.5] * n
        ranks = _average_ranks(values)
        return [r / (n - 1) for r in ranks]
    if mode == "normal":
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        if std == 0:
            return [0.0] * n
        return [(v - mean) / std for v in values]
    raise SynevaError(
        f"unknown normalization '{mode}'; use one of {_MODES}"
    )
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/benchmark/test_normalize.py --no-cov -v` (9 passed).

- [ ] **Step 5: Commit** — `git commit -m "feat(benchmark): cross-candidate normalization (absolute/linear/normal/quantile)"`

---

## Task 2: `benchmark()` + `BenchmarkResult` views (`engine.py`)

**Files:** Create `src/syneva/benchmark/engine.py`; Test `tests/unit/benchmark/test_engine.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/benchmark/test_engine.py
import math

import pytest

from syneva.benchmark.engine import BenchmarkResult, benchmark
from syneva.core.errors import SynevaError


def test_runs_each_candidate(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata
    )
    assert isinstance(res, BenchmarkResult)
    assert set(res.reports) == {"good", "shifted"}
    assert "good" in res.score_matrix and "shifted" in res.score_matrix


def test_overall_is_equal_weight_c_mean(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    cs = res.c_scores["good"]
    expected = sum(cs.values()) / len(cs)
    assert math.isclose(res.overall["good"], expected, rel_tol=1e-9)


def test_metrics_and_cdims_are_sorted_unions(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    assert res.metrics() == sorted(res.metrics())
    assert res.c_dims() == sorted(res.c_dims())


def test_empty_candidates_raises(real_df, metadata):
    with pytest.raises(SynevaError, match="at least one candidate"):
        benchmark(real_df, {}, metadata)


def test_schema_mismatch_candidate_raises(real_df, syn_good_df, metadata):
    bad = syn_good_df.drop(columns=[syn_good_df.columns[0]])
    with pytest.raises(SynevaError, match="bad_cand"):
        benchmark(real_df, {"ok": syn_good_df, "bad_cand": bad}, metadata)


def test_preset_forwarded_to_all(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata,
        preset="privacy",
    )
    for name in ("good", "shifted"):
        assert set(res.c_scores[name]) == {"compliance"}
```

(Confirm fixtures `real_df`/`syn_good_df`/`syn_shifted_df`/`metadata` exist in `tests/conftest.py`; if the shifted fixture has a different name, use the real one.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/benchmark/test_engine.py --no-cov -v` (ImportError).

- [ ] **Step 3: Implement** — create `src/syneva/benchmark/engine.py`:

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.core.presets import _UNSET
from syneva.core.report import Report, _normalize_scalars
from syneva.core.runner import evaluate


def benchmark(
    real: pd.DataFrame | None,
    candidates: dict[str, pd.DataFrame],
    metadata: Metadata | None = None,
    *,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    tiers=_UNSET,
    cs=_UNSET,
    run_utility=_UNSET,
    utility_tasks=None,
    run_fairness=_UNSET,
    fairness_specs=None,
    distance=_UNSET,
    random_state: int = 42,
) -> BenchmarkResult:
    if not candidates:
        raise SynevaError("benchmark needs at least one candidate")
    # Validate column sets up front (fail fast, name the offender).
    ref_cols = set(real.columns) if real is not None else set(next(iter(candidates.values())).columns)
    for name, df in candidates.items():
        if set(df.columns) != ref_cols:
            raise SynevaError(f"candidate '{name}' columns must match real")
    reports: dict[str, Report] = {}
    for name, df in candidates.items():
        reports[name] = evaluate(
            real,
            df,
            metadata,
            holdout=holdout,
            preset=preset,
            tiers=tiers,
            cs=cs,
            run_utility=run_utility,
            utility_tasks=utility_tasks,
            run_fairness=run_fairness,
            fairness_specs=fairness_specs,
            distance=distance,
            random_state=random_state,
        )
    return BenchmarkResult(reports=reports, random_state=random_state)


@dataclass
class BenchmarkResult:
    reports: dict[str, Report]
    random_state: int = 42
    notes: list[str] = field(default_factory=list)

    @property
    def score_matrix(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, rep in self.reports.items():
            row: dict[str, float] = {}
            for r in rep.results:
                if r.error is None and r.scalars is not None:
                    row[r.spec.name] = _normalize_scalars(r)
            out[name] = row
        return out

    @property
    def c_scores(self) -> dict[str, dict[str, float]]:
        return {name: rep.aggregated for name, rep in self.reports.items()}

    @property
    def overall(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, cs in self.c_scores.items():
            out[name] = (sum(cs.values()) / len(cs)) if cs else 0.0
        return out

    def metrics(self) -> list[str]:
        names: set[str] = set()
        for row in self.score_matrix.values():
            names.update(row)
        return sorted(names)

    def c_dims(self) -> list[str]:
        dims: set[str] = set()
        for cs in self.c_scores.values():
            dims.update(cs)
        return sorted(dims)

    def _metric_c_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rep in self.reports.values():
            for r in rep.results:
                out[r.spec.name] = r.spec.c
        return out
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/benchmark/test_engine.py --no-cov -v` (6 passed).

- [ ] **Step 5: Commit** — `git commit -m "feat(benchmark): benchmark() engine + BenchmarkResult views"`

---

## Task 3: ranking, serialization, public export, integration

**Files:** Modify `src/syneva/benchmark/engine.py`, `src/syneva/benchmark/__init__.py`, `src/syneva/__init__.py`; Test append `tests/unit/benchmark/test_engine.py`, create `tests/integration/test_benchmark.py`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/benchmark/test_engine.py
def test_ranking_better_candidate_wins_all_modes(
    real_df, syn_good_df, syn_shifted_df, metadata
):
    res = benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata
    )
    for mode in ("absolute", "linear", "normal", "quantile"):
        ranked = res.ranking(normalization=mode)
        assert ranked[0][1] == "good", f"mode={mode}"
        assert [r[0] for r in ranked] == [1, 2]


def test_ranking_by_single_c(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata
    )
    ranked = res.ranking(by="compliance")
    assert {r[1] for r in ranked} == {"good", "shifted"}


def test_ranking_unknown_by_raises(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    with pytest.raises(SynevaError, match="by"):
        res.ranking(by="not_a_thing")


def test_single_candidate_ranks_first_with_note(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"only": syn_good_df}, metadata)
    ranked = res.ranking(normalization="linear")
    assert ranked == [(1, "only", ranked[0][2])]


def test_roundtrip_dict_preserves_ranking(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata
    )
    res2 = BenchmarkResult.from_dict(res.to_dict())
    assert res2.ranking() == res.ranking()
    assert res2.overall == res.overall
```

```python
# tests/integration/test_benchmark.py
import syneva


def test_benchmark_via_top_level(real_df, syn_good_df, syn_shifted_df, metadata):
    res = syneva.benchmark(
        real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata
    )
    ranked = res.ranking()
    assert ranked[0][1] == "good"
    assert isinstance(res, syneva.BenchmarkResult)
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/benchmark/test_engine.py -k "ranking or roundtrip or single_candidate" tests/integration/test_benchmark.py --no-cov -v`.

- [ ] **Step 3: Implement ranking + serialization** — add these methods to `BenchmarkResult` (after `_metric_c_map`):

```python
    def _normalized_per_metric(self, normalization: str) -> dict[str, dict[str, float]]:
        from syneva.benchmark.normalize import normalize_across

        sm = self.score_matrix
        out: dict[str, dict[str, float]] = {}
        for m in self.metrics():
            present = [c for c in self.reports if m in sm[c]]
            vals = [sm[c][m] for c in present]
            normed = normalize_across(vals, normalization)
            out[m] = dict(zip(present, normed))
        return out

    def _candidate_key(
        self, candidate: str, metric_names: list[str], nm: dict[str, dict[str, float]]
    ) -> float:
        cmap = self._metric_c_map()
        by_c: dict[str, list[float]] = defaultdict(list)
        for m in metric_names:
            v = nm.get(m, {}).get(candidate)
            if v is not None:
                by_c[cmap[m]].append(v)
        if not by_c:
            return 0.0
        c_means = [sum(vs) / len(vs) for vs in by_c.values()]
        return sum(c_means) / len(c_means)

    def ranking(
        self, normalization: str = "absolute", by: str = "overall"
    ) -> list[tuple[int, str, float]]:
        cands = list(self.reports)
        if len(cands) < 2 and normalization != "absolute" and (
            "single-candidate cohort: relative normalization is degenerate"
            not in self.notes
        ):
            self.notes.append(
                "single-candidate cohort: relative normalization is degenerate"
            )
        nm = self._normalized_per_metric(normalization)
        cmap = self._metric_c_map()
        if by == "overall":
            metric_names = self.metrics()
        elif by in self.c_dims():
            metric_names = [m for m in self.metrics() if cmap.get(m) == by]
        elif by in self.metrics():
            metric_names = [by]
        else:
            raise SynevaError(
                f"unknown ranking target by='{by}'; use 'overall', a C dimension, or a metric name"
            )
        keys = {c: self._candidate_key(c, metric_names, nm) for c in cands}
        order = sorted(cands, key=lambda c: (-keys[c], c))
        return [(i + 1, c, keys[c]) for i, c in enumerate(order)]

    def to_dict(self) -> dict:
        return {
            "reports": {name: rep.to_dict() for name, rep in self.reports.items()},
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        return cls(
            reports={name: Report.from_dict(r) for name, r in d["reports"].items()},
            random_state=d.get("random_state", 42),
        )

    def to_json(self, path) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_json(cls, path) -> BenchmarkResult:
        import json
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text()))
```

- [ ] **Step 4: Public export** — in `src/syneva/benchmark/__init__.py`:
```python
from syneva.benchmark.engine import BenchmarkResult, benchmark

__all__ = ["BenchmarkResult", "benchmark"]
```
In `src/syneva/__init__.py`, add the import (after the `from syneva.core.runner import evaluate` line):
```python
from syneva.benchmark import BenchmarkResult, benchmark
```
and add `"BenchmarkResult"` and `"benchmark"` to `__all__` (keep it sorted: `BenchmarkResult` after `ColumnType`? — place alphabetically; ruff/convention: insert `"BenchmarkResult",` before `"ColumnMetadata",` and `"benchmark",` before `"evaluate",`).

CAUTION: `syneva/benchmark/` is a new SUBPACKAGE named `benchmark`, and `benchmark` is also the function. The package `__init__.py` re-exports the function from `engine`, so `from syneva.benchmark import benchmark` resolves to the function. In `syneva/__init__.py`, `from syneva.benchmark import BenchmarkResult, benchmark` then binds `syneva.benchmark` to the FUNCTION (shadowing the submodule name at the top-level namespace) — this is intended so `syneva.benchmark(...)` works. Verify `import syneva; syneva.benchmark` is callable and `syneva.BenchmarkResult` exists.

- [ ] **Step 5: Run, expect pass** — `uv run pytest tests/unit/benchmark/ tests/integration/test_benchmark.py --no-cov -v` (all pass).

- [ ] **Step 6: FULL SUITE (behavior-preservation gate)** — `uv run pytest -q`. ALL pass (2 pre-existing skips allowed); golden snapshot UNCHANGED (NEVER `--update-golden`); coverage >= 85%. If golden fails, STOP and report BLOCKED.

- [ ] **Step 7: Commit** — `git commit -m "feat(benchmark): ranking, serialization, public benchmark() export"`

---

## Self-review notes
- **Spec coverage:** normalization modes (T1) · benchmark() + validation + score_matrix/c_scores/overall/metrics/c_dims (T2) · ranking with normalization+by, degenerate-cohort note, serialization round-trip, public re-export, integration + golden gate (T3).
- **Type/signature consistency:** `benchmark(real, candidates, metadata, *, holdout, preset, tiers, cs, run_utility, utility_tasks, run_fairness, fairness_specs, distance, random_state)` mirrors `evaluate`'s knobs and forwards `_UNSET` unchanged; `normalize_across(values, mode)` used by `_normalized_per_metric`; `ranking()` returns `[(rank, name, key)]`; `to_dict`/`from_dict` mirror `Report`. `c_dims()` named to avoid the `cs=` param clash (per spec).
- **Behavior preservation:** purely additive (new package + 2 export lines); no metric/runner edits; golden checked full-suite in T3.
- **No placeholders:** every step has complete code.
