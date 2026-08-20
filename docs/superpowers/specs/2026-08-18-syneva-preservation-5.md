# syneva — Preservation dimension (novelty #1) — Design

**Date:** 2026-08-18
**Status:** Approved (pending spec review)
**Depends on:** C1–C4 on `main`. Public API: `registry`, `MetricSpec`, `MetricResult`, `Metadata`,
`encode_pair`, `evaluate`/`evaluate_with`, the `metric_info` registry, `UtilityTask`, the Gower
distance backend, holdout support, and the C4 benchmark passthrough.
**Roadmap:** component C5 in `2026-06-23-syneva-master-roadmap.md`. Resolves the C5 open question
(line 187): preservation is its **own dimension**, not a Coverage extension.

## Goal

Add the **preservation** dimension — headline novelty #1: does the synthetic data preserve rare
categories, distribution tails, and small subgroups of the real data? Six metrics: three
auto-detected (no config) and three driven by a new `SubgroupSpec` config. All metrics are
bespoke lightweight statistics (no model panels, no re-run of other dimensions) — robust on the
very small row counts that minorities have, and easy to unit-test.

## Scope

**In scope (6 metrics, all `tier="extended"`, `scope="table-level"`, `data_types={"static"}`,
`requires_real=True`; every `score` is in [0, 1] with 1 = ideal):**

| name | needs | score (1 = ideal) |
|---|---|---|
| `rare_category_retention` | categorical cols | mean retention of rare categories |
| `tail_coverage` | numeric cols | mean synthetic mass in real tails |
| `minority_class_density` | categorical cols | mean symmetric density ratio of minority classes |
| `subgroup_fidelity` | `SubgroupSpec` | mean of ½·shape + ½·share per subgroup |
| `minority_utility_gap` | `SubgroupSpec` + `UtilityTask` | 1 − excess TSTR gap vs TRTR baseline |
| `minority_privacy_risk` | `SubgroupSpec` | worst-subgroup DCR concentration ratio |

Plus: the new `preservation` dimension, `SubgroupSpec` + runner plumbing, metric-info entries,
a minimal UI subgroup builder.

**Out of scope (later):** CLI flags for subgroup specs (reachable via `--cs preservation` for the
auto-metrics; explicit flags deferred, same as fairness); auto-*suggestion* of subgroups
(`suggest_subgroups(meta)` analog); longitudinal/relational variants; paper experiments (C9).

## New dimension

Add `"preservation"` to the `C` Literal in `src/syneva/core/metric.py` and
`C_INFO["preservation"]` in `src/syneva/core/metric_info.py`:
`"preservation": "Does the synthetic data preserve rare categories, distribution tails, and small subgroups of the real data?"`.
Runner, aggregation, renderer, benchmark, and UI group by `spec.c`, so the card, leaderboard
column, and `ranking(by="preservation")` appear automatically.

## `SubgroupSpec`

New package `src/syneva/preservation/` (mirrors `fairness/`): `spec.py` + one module per metric +
`__init__.py` side-effect imports, registered from `syneva/__init__.py` (re-export `SubgroupSpec`).

```python
@dataclass
class SubgroupSpec:
    name: str                                # display key, used in per_column
    conditions: dict[str, list | tuple]      # AND-ed; list = categorical membership,
                                             # (lo, hi) tuple = numeric range, None = open end
```

- `matches(df: pd.DataFrame) -> pd.Series` (boolean mask) — the single shared row-selection
  implementation all three subgroup metrics use. Range bounds are inclusive.
- Validation at match time: unknown column, a range on a categorical column, or a values-list on a
  numeric column → the whole spec is skipped with a note naming the spec and the problem.

## Gating (differs from fairness — deliberate)

**No `run_preservation` flag.** Fairness is gated because *all* its metrics are meaningless
without config; here three metrics are config-free and cheap, so the dimension is on by default
for any run whose tiers include `extended`.

- `evaluate()`/`evaluate_with()` gain `subgroup_specs: list[SubgroupSpec] | None = None`.
- `_instantiate` injects `subgroup_specs` by constructor-param inspection (same mechanism as
  `tasks`/`specs`/`holdout`/`distance`/`random_state`).
- The runner **drops** (does not instantiate) any selected metric whose constructor has a
  `subgroup_specs` param when `subgroup_specs` is `None` or empty — default runs show the three
  auto-metrics with zero skipped-metric noise.
- Presets untouched: `full` picks the dimension up via the `extended` tier; no new `Preset` field.
- `benchmark()` forwards `subgroup_specs` to `evaluate` like it forwards `fairness_specs`.
- Golden snapshot untouched: golden runs `tiers=("core",)` and every preservation metric is
  `extended`.

## Metric designs

Constants: `_EPS = 1e-12`; all scores clamp to [0, 1]; every threshold below is a constructor
field with the stated default (tunable, seeded where sampling occurs, `random_state: int = 42`).

### 1. `rare_category_retention` (auto)
- **Measures:** whether categories that are rare in the real data survive into the synthetic data.
- **Config:** `rare_threshold: float = 0.05` — a category is *rare* when its real relative
  frequency is `< rare_threshold`.
- **Compute:** per categorical column, for each rare category with real frequency `p_real` and
  synthetic frequency `p_syn`: `retention = min(p_syn / p_real, 1.0)` (under-representation only;
  over-representation is scored by `minority_class_density` and congruence).
  `score = mean over all rare categories across all categorical columns`.
- **Scalars:** `{score, n_rare_categories, pct_rare_missing}` (`pct_rare_missing` = share of rare
  categories with `p_syn == 0`). `per_column[col] = mean retention` for each categorical column
  that has rare categories.
- **Skips:** no categorical columns, or no rare categories anywhere → `score = 1.0` + note.

### 2. `tail_coverage` (auto)
- **Measures:** whether the synthetic data reaches into the tails of each real numeric distribution.
- **Config:** `tail_quantile: float = 0.05`.
- **Compute:** per numeric column, tail cutoffs are the real `q` and `1 − q` quantiles. Lower-tail
  coverage = `min(share of synthetic values < q-cutoff / tail_quantile, 1.0)`; upper tail
  analogous (strictly-beyond on the real cutoffs, consistent both sides).
  `score = mean over both tails of all numeric columns`.
- **Scalars:** `{score, lower_tail_coverage, upper_tail_coverage}` (means across columns).
  `per_column[col] = mean of the column's two tail coverages`.
- **Skips:** no numeric columns → `score = 1.0` + note. A constant real column (degenerate
  quantiles) → that column skipped + note. A tail whose cutoff equals the real column's
  extremum (e.g. a bounded/discrete column where `lo == min` or `hi == max`) is strictly
  impossible and is skipped independently of the other tail, with a note naming the column.

### 3. `minority_class_density` (auto)
- **Measures:** whether each categorical column's single least-frequent class keeps its density.
  Complements #1: no rarity threshold, and over-representation is penalized too.
- **Compute:** per categorical column, minority class = the least-frequent category in the real
  column (ties → first by sorted label, deterministic).
  `density_ratio = min(p_syn, p_real) / max(p_syn, p_real)` (0 when the class vanishes).
  `score = mean over categorical columns`.
- **Scalars:** `{score, worst_density_ratio}`. `per_column[col] = density_ratio`.
- **Skips:** no categorical columns → `score = 1.0` + note. A single-category column contributes
  ratio via its (only) class as normal.

### 4. `subgroup_fidelity` (needs `SubgroupSpec`)
- **Measures:** whether each declared subgroup keeps its size and its internal distributions.
- **Config:** `subgroup_specs: list[SubgroupSpec]` (runner-injected), `min_rows: int = 10`.
- **Compute, per spec:** masks via `spec.matches` on real and synthetic.
  `share_score = min(share_syn, share_real) / max(share_syn, share_real)` where `share` = subgroup
  fraction of rows. `shape_score = 1 − mean per-column distance` **within** the subgroup rows:
  two-sample KS statistic for numeric columns, total-variation distance for categorical columns
  (both already in [0, 1]). `score_spec = 0.5 * shape_score + 0.5 * share_score`.
- **Edge semantics:** subgroup empty in synthetic → `score_spec = 0.0` + note (the subgroup was
  erased — a finding, not a skip). Real subgroup `< min_rows` → skip spec + note. Invalid spec →
  skip + note. `score = mean over runnable specs`; no runnable specs → `score = 1.0` + note.
- **Scalars:** `{score, worst_subgroup_score, mean_share_drift}`
  (`share_drift = |share_syn − share_real|`). `per_column[spec.name] = {share_real, share_syn,
  shape_score, score}` per spec.

### 5. `minority_utility_gap` (needs `SubgroupSpec` + tasks)
- **Measures:** whether a model trained on synthetic data serves the subgroup as well as it serves
  everyone — beyond the gap that already exists when training on real data.
- **Config:** `subgroup_specs`, `tasks: list[UtilityTask]` (both runner-injected, same as the
  utility metrics receive), `holdout` (injected when provided), `min_rows: int = 10`,
  `random_state: int = 42`.
- **Model:** one lightweight sklearn pipeline per training frame — one-hot (categoricals) +
  standard-scale (numerics) + `LogisticRegression(max_iter=1000)`. Classification targets only;
  numeric-target tasks skipped + note. Metric: balanced accuracy.
- **Compute, per (task × spec):** evaluation frame = `holdout` if provided, else a seeded 50/50
  split of real (train half / eval half; the synthetic-trained model also evaluates on that same
  eval half for comparability). `gap_syn = bacc_overall − bacc_subgroup` for the model trained on
  synthetic; `gap_real` = the same for the model trained on real (the train half, or full real
  when holdout exists). `excess = max(0, gap_syn − gap_real)`;
  `score_pair = clamp(1 − excess)`. `score = mean over runnable (task × spec) pairs`.
- **Skips:** no tasks → `score = 1.0` + note (metric still runs when subgroup_specs exist but
  tasks were never configured/suggested). Subgroup eval rows `< min_rows`, or target constant in
  train or subgroup eval rows → skip pair + note. No runnable pairs → `score = 1.0` + note.
- **Scalars:** `{score, worst_excess_gap, mean_gap_synthetic, mean_gap_real}`.
  `per_column[f"{spec.name}|{task.target}"] = {gap_synthetic, gap_real, excess_gap}`.

### 6. `minority_privacy_risk` (needs `SubgroupSpec`)
- **Measures:** whether disclosure risk concentrates on the subgroup — are its members closer to
  synthetic records than the average real record is?
- **Config:** `subgroup_specs`, `distance: str = "euclidean"` (runner-injected; `"gower"`
  supported), `min_rows: int = 10`, `cap: int = 2000`, `random_state: int = 42`.
- **Compute:** encode real + synthetic in the shared space (`encode_pair`, or the Gower matrix
  when `distance="gower"`). Synthetic index capped at `cap` seeded rows. Real query rows capped at
  `cap` by keeping at most `cap // 2` subgroup members (seeded random choice when there are more)
  and filling the remaining budget with non-members (all of them if fewer than the budget); note
  when capped. If there are no non-member rows left to fill with, the split baseline can't be
  computed — note "subgroup '<name>' risk not measurable (subgroup spans all kept rows)" per spec
  and skip affected specs (score contribution omitted). DCR = distance from each kept real row to
  its nearest synthetic row. Per spec:
  `risk_ratio = median DCR(subgroup members) / (median DCR(all kept rows) + _EPS)`;
  `score_spec = 1.0 if risk_ratio >= 1 else risk_ratio`.
- **Aggregation:** `score = min over runnable specs` (worst subgroup — conservative, the privacy
  convention). Mean reported as a scalar.
- **Skips:** subgroup members among kept rows `< min_rows` → skip spec + note; no runnable specs →
  `score = 1.0` + note.
- **Scalars:** `{score, mean_risk_ratio, worst_risk_ratio}`.
  `per_column[spec.name] = {risk_ratio, median_dcr_subgroup, median_dcr_overall}`.

## metric-info entries (required; guard tests enforce)

For all six metrics: `METRIC_NAMES` ("Rare-category retention", "Tail coverage", "Minority-class
density", "Subgroup fidelity", "Minority utility gap", "Minority privacy risk"), `METRIC_INFO`
(neutral, no direction claim), `RAW_HINT` (scale + direction), and `_SCALAR_LABELS` for every new
scalar key listed above. Plus `C_INFO["preservation"]`.

## UI (minimal, mirrors the fairness section)

- Add `"preservation"` to the hardcoded dimension list in `src/syneva/ui/app.py` (line ~88).
- New sidebar section "Preservation": one optional subgroup builder — name text input, column
  multiselect, per-column value multiselect (categorical) or lo/hi number inputs (numeric) —
  building a single `SubgroupSpec` passed as `subgroup_specs=[spec]`.
- If preservation subgroup metrics are ticked but no subgroup is defined, show an **info** note
  that only the auto-metrics will run (not a blocking warning — the auto-metrics are still valid).
- `src/syneva/ui/core.run_report` gains a `subgroup_specs` param forwarded to `evaluate_with`.

## Testing

TDD per metric with constructed fixtures:
- **rare_category_retention:** synthetic that drops a rare category → low score +
  `pct_rare_missing > 0`; faithful synthetic → ≈ 1; no-rare-categories data → 1.0 + note.
- **tail_coverage:** synthetic truncated at the real p10/p90 → low score; faithful → ≈ 1;
  constant column skipped + note.
- **minority_class_density:** minority class halved → ratio ≈ 0.5; oversampled 2× → ≈ 0.5
  (symmetry); vanished → column ratio 0.
- **subgroup_fidelity:** subgroup erased in synthetic → `score_spec = 0` + note; subgroup shrunk →
  share term drops; within-subgroup distribution shifted → shape term drops; tiny real subgroup →
  skip + note; invalid spec (range on categorical) → skip + note.
- **minority_utility_gap:** constructed data where the synthetic flips the label only inside the
  subgroup → large `excess`, low score; synthetic preserving the relationship → ≈ 1; no tasks →
  1.0 + note.
- **minority_privacy_risk:** synthetic that copies subgroup rows verbatim but not others →
  `risk_ratio ≪ 1`, low score; uniform closeness → ≈ 1. Gower path smoke-tested.
- **Wiring** (`tests/integration/test_preservation_wiring.py`): with `tiers=("core","extended")`
  and no `subgroup_specs`, exactly the three auto-metrics of the dimension run (subgroup metrics
  absent, not skipped); with `subgroup_specs=[...]` (+ tasks) all six run; `tiers=("core",)` runs
  none.
- Metric-info guard tests pass automatically once the entries exist; golden snapshot untouched
  (extended tier); coverage gate ≥ 85 %; parity manifest untouched (novel metrics, not SynthEval).

Branch: `feat/preservation-5`. On merge: update the master roadmap C5 row to ✅ and its open
question as resolved.

## Open questions

None outstanding — placement (own dimension), the six-metric set, `SubgroupSpec` shape
(conditions + ranges), no-flag gating, bespoke-statistics approach, and all thresholds
(rare 0.05, tail 0.05, min_rows 10, ½·shape + ½·share, TRTR-baselined excess gap, worst-subgroup
privacy aggregation) were fixed in brainstorming on 2026-08-18.
