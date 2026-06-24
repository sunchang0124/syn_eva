# syneva — Component C4-a: benchmark engine core — Design

**Date:** 2026-06-24
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + C1–C3 (parity complete). Public API: `evaluate`, `Report`
(`.results`, `.by_c`, `.aggregated`, `.to_dict`/`from_dict`), `Metadata`, `SynevaError`, the `_UNSET`
sentinel + preset/holdout plumbing.
**Roadmap:** component C4 (benchmark & ranking engine), sub-spec C4-a (engine core). Siblings: C4-b
leaderboard report + CLI, C4-c UI benchmark tab.

## Goal

Given one real dataset and N named synthetic candidates, evaluate each through the existing
`evaluate()` with a single shared config and produce a serializable `BenchmarkResult` that ranks the
candidates. Displayed per-C and overall scores are the absolute, cohort-independent [0,1] metric
scores; a selectable normalization mode (`absolute`/`linear`/`normal`/`quantile`) controls the ranking
order only. No rendering, CLI, or UI here (C4-b/c) — but the result fully serializes so they can
consume it.

## Scope

**In scope:** new `src/syneva/benchmark/` package with a `benchmark(...)` entry point, a
`BenchmarkResult` dataclass (score matrix, per-C scores, overall, ranking with normalization),
cross-candidate normalization math, schema validation, serialization, the public re-export.

**Out of scope:** leaderboard HTML/JSON rendering, CLI `benchmark` command, UI tab (C4-b/c); learned/
calibrated per-C weighting (C6); accepting pre-computed Reports (YAGNI).

## Architecture

New package `src/syneva/benchmark/` (`__init__.py`, `engine.py`, `normalize.py`).

### Entry point — `benchmark(...)` (in `engine.py`)

```python
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
```

- `candidates` maps a display name → synthetic DataFrame. Empty dict → `SynevaError`.
- Up-front validation: for each `name, df`, if `real is not None` and `set(df.columns) !=
  set(real.columns)` → `SynevaError(f"candidate '{name}' columns must match real")` (fail fast,
  before running anything). (When `real is None`, syneva still supports real-free metrics; validate
  candidate column sets are mutually consistent against the first candidate instead.)
- For each candidate, call `evaluate(real, df, metadata, holdout=holdout, preset=preset, tiers=tiers,
  cs=cs, run_utility=run_utility, utility_tasks=utility_tasks, run_fairness=run_fairness,
  fairness_specs=fairness_specs, distance=distance, random_state=random_state)` — the SAME config for
  all, so comparisons are apples-to-apples and reproducible (`_UNSET` values flow through unchanged,
  preserving preset/default resolution). Collect `reports: dict[name, Report]`, preserving insertion
  order.
- Metric-level failures inside a candidate remain captured in `MetricResult.error`; `Report.aggregated`
  already filters them out. A candidate is never dropped for a metric error; only a schema mismatch
  fails the whole run (deliberate — schema mismatch is a usage error).

### `BenchmarkResult` dataclass

Fields: `reports: dict[str, Report]`, `random_state: int`. Computed via properties/cached on the
absolute [0,1] scores:

- `score_matrix -> dict[str, dict[str, float]]`: `{candidate: {metric_name: score}}`, where score is
  the metric's normalized [0,1] value (1 = ideal) obtained from each result the same way
  `Report.aggregated` does (reuse the existing `_normalize_scalars` helper from `core.report`;
  promote/import it rather than reimplementing). Metric results with `error` or no scalars are omitted
  for that candidate.
- `c_scores -> dict[str, dict[str, float]]`: `{candidate: report.aggregated}` (per-C mean of the
  metric scores) — directly reuses `Report.aggregated`.
- `overall -> dict[str, float]`: per candidate, the equal-weight mean across that candidate's active
  Cs (`mean(c_scores[candidate].values())`); `0.0` if a candidate has no usable Cs.
- `metrics() -> list[str]`: the sorted union of metric names across candidates (stable column order).
- `c_dims() -> list[str]`: the sorted union of C dimensions present. (Named `c_dims` to avoid
  confusion with the `cs=` filter parameter of `benchmark()`.)
- `ranking(normalization: str = "absolute", by: str = "overall") -> list[tuple[int, str, float]]`:
  ordered `[(rank, candidate, ranking_score)]`, rank 1 = best, sorted by `ranking_score` descending
  (ties broken by candidate name for determinism).

### Ranking & normalization (`normalize.py`)

The ranking key per candidate is built from the absolute score matrix, transformed by the chosen
normalization **per metric across candidates**, then aggregated to the `by` target:

1. Select the metric set for `by`:
   - `by == "overall"` → all metrics, aggregated as mean-per-C then mean-across-Cs (equal C weight, so
     a C with many metrics doesn't dominate).
   - `by == <c>` (a C dimension) → only that C's metrics, mean.
   - `by == <metric_name>` → that single metric.
   - Unknown `by` → `SynevaError`.
2. Per metric, gather the candidates' absolute scores and transform across the cohort:
   - `"absolute"`: identity (the score itself).
   - `"linear"`: min-max to [0,1]: `(s - lo) / (hi - lo)`; if `hi == lo` → `1.0` for all (constant
     metric, note added).
   - `"quantile"`: rank candidates ascending (average ranks for ties) → `rank / (n - 1)` in [0,1]; if
     `n < 2` → `0.5`.
   - `"normal"`: z-score `(s - mean) / std`; if `std == 0` → `0.0` for all. (Ranking key may be any
     real; it is relative, not displayed.)
   - Unknown normalization → `SynevaError`.
3. Aggregate the per-metric transformed values to the `by` target (means as in step 1) → the
   candidate's `ranking_score`.

`n < 2` (single candidate) → every relative mode degenerates: ranking still returns the lone candidate
at rank 1; a note is recorded on the result (`notes: list[str]`). Displayed `score_matrix`/`c_scores`/
`overall` are ALWAYS the absolute values, independent of `normalization`.

### Serialization

`BenchmarkResult.to_dict()` →
```python
{
  "reports": {name: report.to_dict() for name, report in reports.items()},
  "random_state": random_state,
}
```
`BenchmarkResult.from_dict(d)` rebuilds `reports` via `Report.from_dict` and restores `random_state`;
the score matrix / rankings are recomputed lazily from the reports (not persisted, so they can't drift
from the underlying results). `to_json(path)` / `from_json(path)` thin wrappers mirror `Report`.

### Public re-export

Expose `benchmark` and `BenchmarkResult` from the top-level `syneva` package (`syneva.benchmark(...)`),
matching how `evaluate` is exported.

## Error handling

- Empty `candidates` → `SynevaError("benchmark needs at least one candidate")`.
- Candidate schema mismatch → `SynevaError` naming the candidate (fail fast).
- Unknown `normalization` or `by` in `ranking()` → `SynevaError`.
- A candidate whose every metric errors still produces a `Report` with `overall == 0.0` and a note; it
  is ranked last, not dropped.

## Testing

- **Runs N candidates:** `benchmark(real, {"a": good, "b": shifted})` returns a `BenchmarkResult` with
  a `Report` per candidate and a `score_matrix` covering both.
- **Overall = equal-weight C mean:** `overall[c]` equals `mean(c_scores[c].values())`.
- **Better candidate wins under every mode:** using the `adult_income` good vs shifted vs leaky
  fixtures, the good candidate ranks #1 for `normalization` in {absolute, linear, normal, quantile}.
- **Normalization math:** on a hand-built score matrix (e.g. metric scores {a:0.2, b:0.8, c:0.5}),
  assert linear → {a:0, b:1, c:0.5}, quantile → {a:0, b:1, c:0.5}, normal → z-scores summing to ~0,
  absolute → unchanged.
- **Degenerate cohorts:** single candidate → rank 1 + note; a metric constant across candidates →
  linear 1.0 / normal 0.0 fallback, no crash/NaN.
- **`by` targets:** `ranking(by="compliance")` ranks on compliance only; `ranking(by="<metric>")`
  on one metric; unknown `by` raises.
- **Shared config:** preset/holdout/tiers forwarded identically (e.g. `preset="privacy"` yields only
  compliance metrics for every candidate); schema-mismatch candidate raises `SynevaError` naming it.
- **Round-trip:** `to_dict`/`from_dict` (and `to_json`/`from_json`) reproduce the same rankings.
- 85% coverage gate holds.

## Open questions
None outstanding. Normalization affects ranking order only; displayed scores remain absolute. Overall
is an equal-weight mean across Cs (learned weighting deferred to C6).
