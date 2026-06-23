# syneva — SynthEval parity, Phase 0b: privacy + Fairness — Design

**Date:** 2026-06-23
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + Phase 0a. Public API: `registry`, `MetricSpec`, `MetricResult`, `Metadata`/`ColumnMetadata` (with the `sensitive` flag), `ColumnType`, `encode_pair`, `evaluate`/`evaluate_with`, the `metric_info` registry, and the established `utility_tasks`/`run_utility` config pattern.
**Roadmap:** component C2 in `2026-06-23-syneva-master-roadmap.md`.

## Goal

Add the four SynthEval metrics syneva still lacks — three privacy/disclosure metrics and one fairness
metric — completing 21/21 SynthEval metric coverage. Introduce a new **Fairness** dimension and a
`FairnessSpec` config that mirrors the existing utility-task pattern. Update the parity manifest so the
CI test then proves full SynthEval coverage.

## Scope

**In scope (4 metrics):**

| name | dimension / tier | scope | needs | score (1 = ideal) |
|---|---|---|---|---|
| `hitting_rate` | compliance / core | table-level | real only | `1 − hit_rate` |
| `epsilon_identifiability` | compliance / extended | table-level | real only | `1 − risk` |
| `attribute_disclosure` | compliance / extended | table-level | `sensitive=True` cols | `1 − disclosure_rate` |
| `statistical_parity` | **fairness** / extended | table-level | `FairnessSpec` | `1 − parity_drift` |

Plus: the new `fairness` dimension, the `FairnessSpec` config + runner/UI plumbing, metric-info entries,
and the parity-manifest update.

**Out of scope (later phases):** Gower distance, holdout support, presets, multi-classifier utility (0c);
entropy-weighted identifiability (a refinement, noted below); benchmark/ranking (C4).

## New dimension

Add `"fairness"` to the `C` Literal in `src/syneva/core/metric.py` and a `C_INFO["fairness"]` entry in
`src/syneva/core/metric_info.py`:
`"fairness": "Does the synthetic data preserve the real data's fairness across protected groups?"`.
The runner, aggregation, renderer, and UI already group by `spec.c`, so a fairness card appears
automatically.

## Metric designs

`_EPS = 1e-12`; scores clamp to `[0, 1]`; all four are `requires_real=True`, `scope="table-level"`.

### 1. `hitting_rate` (compliance, core)
- **Measures:** how often the generator places a synthetic record right on top of a real one.
- **Threshold:** per numeric column, `thresh = (max − min) / 30` computed on the **real** column (the
  SynthEval 1/30 default); categorical columns require an exact match.
- **Compute:** a real record is "hit" if there exists a synthetic record within `thresh` on every numeric
  column and equal on every categorical column. `hit_rate = (# real records hit) / n_real`. For cost, cap
  both frames at 2000 rows (seeded `np.random.default_rng(42)` subsample) and note when capped.
- **score:** `1 − hit_rate`. scalars `{score, hit_rate}`. Raw direction: lower is better.

### 2. `epsilon_identifiability` (compliance, extended)
- **Measures:** identifiability risk (Yoon et al.): is a real record closer to a synthetic record than to
  its own nearest real neighbour?
- **Compute:** `X_real, X_syn = encode_pair(real, synthetic, meta)`. `d_self` = distance from each real
  record to its nearest *other* real record (`NearestNeighbors(2)`, column 1). `d_syn` = distance to the
  nearest synthetic record (`NearestNeighbors(1)`). `identified_i = d_syn[i] < d_self[i]`.
  `risk = mean(identified)`.
- **score:** `1 − risk`. scalars `{score, identifiability_risk}`. Raw direction: lower is better.
- **Note (future refinement, not v1):** SynthEval weights features by entropy before the distance; v1 uses
  the unweighted shared-space distance. Recorded as a 0c/refinement item; documented in the metric's note.

### 3. `attribute_disclosure` (compliance, extended)
- **Measures:** can an attacker infer a sensitive attribute from the quasi-identifiers via a nearest
  synthetic record?
- **Config:** sensitive columns `S = [n for n, cm in meta.columns if cm.sensitive]`; quasi-identifiers
  `QI = [n for ...　if not sensitive and dtype in (numeric, categorical, boolean)]`. If `S` is empty or
  `QI` is empty → return `score=1.0` with a note (mirrors `k_anonymity`'s skip-on-no-sensitive behavior).
- **Compute:** build a QI-restricted `Metadata` (`qi_meta = Metadata(columns={n: meta.columns[n] for n
  in QI})`) and encode the QI columns of both frames in a shared space via
  `encode_pair(real, synthetic, qi_meta)` (encode_pair already selects only the columns present in the
  passed metadata). For each real record find its nearest synthetic record by QI distance; the attacker
  "guesses" that synthetic record's sensitive values. A guess
  is correct if every sensitive value matches — categorical sensitive exact; numeric sensitive within
  `(max−min)/30` of the real value. `disclosure_rate = mean(correct)`.
- **score:** `1 − disclosure_rate`. scalars `{score, disclosure_rate}`. Raw direction: lower is better.

### 4. `statistical_parity` (fairness, extended)
- **Config:** one or more `FairnessSpec(protected_attribute: str, outcome: str, favorable_outcome=None)`.
  `favorable_outcome` default: when `None`, use `max(unique(outcome))` (gives `1` for 0/1 and `True` for
  booleans; for multiclass it selects the top label, with a note).
- **Per spec, on a dataframe:** groups = unique values of `protected_attribute`;
  `rate_g = P(outcome == favorable | protected_attribute == g)`; `SPD = max_g rate_g − min_g rate_g`.
  Compute `spd_real` on real and `spd_synthetic` on synthetic; `parity_drift = |spd_synthetic − spd_real|`.
- **Across specs:** `score = mean over specs of clamp(1 − parity_drift)`; `per_column[f"{A}|{Y}"] =
  {spd_synthetic, spd_real, parity_drift}`; top-level scalars `{score, spd_synthetic, spd_real,
  parity_drift}` report the first spec (or the mean) for the headline.
- **Skips:** if `protected_attribute` or `outcome` is missing from the columns → note and skip that spec;
  if no runnable specs → `score=1.0` + note.
- **Raw direction:** `parity_drift` and `spd_*` — lower is better (0 = synthetic preserves real parity).

## Config plumbing (mirror the utility pattern exactly)

- New `FairnessSpec` dataclass in `src/syneva/fairness/spec.py` (new `fairness/` package):
  `@dataclass class FairnessSpec: protected_attribute: str; outcome: str; favorable_outcome: object | None = None`.
- `evaluate()` / `evaluate_with()` gain `fairness_specs: list[FairnessSpec] | None = None` and
  `run_fairness: bool = False`. Like utility: `if not run_fairness: selected = [c for c in selected if
  c.spec.c != "fairness"]`. (No auto-suggest — the protected attribute must be declared deliberately; if
  `run_fairness=True` but `fairness_specs is None`, fairness metrics run and self-skip with a note.)
- Generalize the runner's `_instantiate(cls, ...)` so it passes `specs=fairness_specs` to any metric
  whose constructor accepts a `specs` parameter (alongside the existing `tasks`/`random_state` handling).
  `StatisticalParity` is a dataclass with `specs: list[FairnessSpec] = field(default_factory=list)` and
  `random_state: int = 42`.
- Re-export `FairnessSpec` from `syneva/__init__.py`.

## UI (minimal, mirrors utility-tasks UI)

In `src/syneva/ui/app.py` `_sidebar()`: when at least one `fairness` metric is ticked, show a "Fairness"
section with a protected-attribute selectbox and an outcome selectbox (both over the uploaded columns);
build a single `FairnessSpec` and pass `fairness_specs=[spec]`, `run_fairness=True` into
`core.run_report(...)`. `src/syneva/ui/core.run_report` gains `fairness_specs`/`run_fairness` params
forwarded to `evaluate_with` (run_fairness inferred = any selected metric has `c=="fairness"`, paralleling
the existing utility handling). If a fairness metric is selected but no protected attribute/outcome chosen,
warn (same shape as the existing "utility selected but no target" guard).

## metric-info entries (required; guard tests enforce)

Add to `src/syneva/core/metric_info.py` for all four metrics: `METRIC_NAMES` (full names: "Hitting rate",
"Epsilon identifiability risk", "Attribute disclosure risk", "Statistical parity difference"),
`METRIC_INFO` (neutral, no direction claim), `RAW_HINT` (all four "lower is better"), and `_SCALAR_LABELS`
for the new scalar keys (`hit_rate`, `identifiability_risk`, `disclosure_rate`, `spd_synthetic`,
`spd_real`, `parity_drift`). Add `C_INFO["fairness"]`.

## Parity manifest

Flip the four entries in `tests/parity/test_syntheval_parity.py` from `("planned", "0b")` to
`("implemented", <metric_name>)`: `hit_rate→hitting_rate`, `eps_risk→epsilon_identifiability`,
`att_discl→attribute_disclosure`, `statistical_parity→statistical_parity`. The existing parity test then
asserts **21/21** SynthEval metrics are registered.

## Testing

TDD per metric:
- **hitting_rate:** `syn_leaky` fixture (80 verbatim rows) → high hit_rate, low score; `syn_good` → lower.
- **epsilon_identifiability:** leaky → high risk/low score; a far-shifted synthetic → low risk/high score.
- **attribute_disclosure:** constructed data where a sensitive column is a deterministic function of a QI
  and the synthetic copies it → high disclosure; an independent synthetic → low. Plus a no-sensitive-column
  dataset → score 1.0 + note.
- **statistical_parity:** constructed real with a known group bias; a synthetic preserving it → drift ≈ 0,
  score ≈ 1; a synthetic that flattens/inverts it → large drift, low score. Plus `adult_income` with
  `FairnessSpec("sex", "high_income")`.
- **Plumbing:** an integration test that `evaluate(real, syn, meta, tiers=("core","extended"),
  run_fairness=True, fairness_specs=[FairnessSpec("sex","high_income")])` includes `statistical_parity`
  with no error, and that fairness metrics are absent when `run_fairness=False`.

The 85% coverage gate must hold. Golden snapshot: `hitting_rate` is **core**, so the `tiers=("core",)`
golden will gain it — regenerate the golden snapshot as part of the work (intentional, like 0a).

## Open questions
None outstanding; the hitting-rate 1/30 threshold and the `favorable_outcome=max(unique)` default are
fixed above.
