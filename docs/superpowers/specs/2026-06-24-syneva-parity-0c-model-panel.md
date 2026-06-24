# syneva — Phase 0c-d: multi-classifier (model-panel) utility — Design

**Date:** 2026-06-24
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + 0a + 0b + 0c-a/b/c. Public API: the metric registry/runner,
`UtilityTask`, `TSTRSuite` (and its `_score`), `build_pipeline`/`select_features`, `Metadata`,
`metric_info` (METRIC_NAMES/METRIC_INFO/_SCALAR_LABELS/RAW_HINT), `holdout`, presets.
**Roadmap:** component C3 (methodology rigor), sub-spec C3d. Final C3 sub-spec; completes C3.

## Goal

Add an extended-tier utility metric that runs the TSTR/TRTR comparison across a **panel of model
families** (linear, random forest, gradient boosting) rather than a single RandomForest, and reports
each family's utility ratio plus their spread. A synthetic dataset that only preserves the signal one
model type exploits is then exposed. The existing core `tstr_suite` (single RF) is unchanged
(behavior-preserving; golden snapshot stays green).

## Why a new metric (not extending `tstr_suite`)

`tstr_suite` is core-tier and drives the `fast` preset; extending it would change every existing
user's headline utility number and force a golden-snapshot regeneration. Instead, `model_panel_utility`
is a NEW extended-tier metric. It runs only under `run_utility=True` (it is `c="utility"`, which the
runner prunes unless utility is requested), so the `full` preset (extended tier + utility) picks it up
while `fast` keeps the quick single-RF. Additive and behavior-preserving.

## Component: model panel in `_models.py`

Add to `src/syneva/utility/_models.py`:

```python
def build_panel_pipelines(
    task_type: str, meta: Metadata, features: list[str], random_state: int
) -> dict[str, Pipeline]:
    """Return {model_name: Pipeline} for the utility model panel. Every pipeline
    shares the SAME preprocessor as build_pipeline; only the final estimator differs."""
```

Three families (classification / regression estimators):
- `"linear"` — `LogisticRegression(max_iter=1000, random_state=random_state)` / `Ridge(random_state=random_state)`
- `"random_forest"` — `RandomForestClassifier/Regressor(n_estimators=100, random_state=random_state, n_jobs=1)`
- `"hist_gbdt"` — `HistGradientBoostingClassifier/Regressor(random_state=random_state)`

The preprocessor is the identical `ColumnTransformer` that `build_pipeline` builds (numeric:
SimpleImputer→StandardScaler; categorical/boolean: SimpleImputer(most_frequent)→OneHotEncoder(
handle_unknown="ignore"); remainder drop). To avoid drift, optionally extract a private
`_build_preprocessor(meta, features)` helper that BOTH `build_pipeline` and `build_panel_pipelines`
call — **only if** the extracted helper produces a byte-identical `ColumnTransformer` so `tstr_suite`
behavior is unchanged. If extraction risks any change, leave `build_pipeline` as-is and duplicate the
preprocessor construction in `build_panel_pipelines`. `build_pipeline` (single RF) is NOT modified in
behavior either way.

`Ridge`, `LogisticRegression`, `HistGradientBoosting*` import from `sklearn.linear_model` and
`sklearn.ensemble`.

## Component: `model_panel_utility` metric

New file `src/syneva/utility/extended/model_panel.py`:

```python
@registry.register
@dataclass
class ModelPanelUtility:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="model_panel_utility",
        c="utility",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )
    tasks: list[UtilityTask] = field(default_factory=list)
    random_state: int = 42
    holdout: object | None = None
```

`compute(real, synthetic, meta)`:
- Reuse `tstr.py`'s `_score` (`from syneva.utility.tstr import _score`) and
  `select_features`/`build_panel_pipelines` from `_models`. Do NOT duplicate `_score` and do NOT
  modify `tstr.py`.
- For each task `t` (skip with a note if `t.target` not in `real.columns`):
  - `features = select_features(meta, t.target, t.features)`; `tt = t.task_type or "regression"`.
  - Build the test set exactly as `TSTRSuite` does: if `self.holdout is not None`, `X_test/y_test` =
    holdout columns and train on full `real`/`synthetic`; else `train_test_split(real[...],
    test_size=0.3, random_state=self.random_state, stratify=real[target] if classification else None)`.
  - `pipes = build_panel_pipelines(tt, meta, features, self.random_state)`.
  - For each `name, _proto in pipes.items()`: build a FRESH pipeline per fit (call
    `build_panel_pipelines` once per (real/syn) need, or rebuild — a fitted pipeline must not be reused
    across train sets). Concretely: `trtr = _score(tt, build_panel_pipelines(...)[name], X_real_tr,
    y_real_tr, X_test, y_test)` and `tstr = _score(tt, build_panel_pipelines(...)[name], X_syn, y_syn,
    X_test, y_test)`. (Implementation may build the dict twice — once for TRTR, once for TSTR — to
    guarantee unfitted estimators; this mirrors how `TSTRSuite` builds a fresh pipeline for each
    `_score` call.)
  - `ratio = clamp(tstr / trtr, 0, 1)` (0.0 if `trtr == 0`).
  - Record `per_column[f"{t.target}/{name}"] = {"trtr": ..., "tstr": ..., "ratio": ...}`.
- If no task ran: return `MetricResult(spec, scalars={"score": 1.0}, notes=[*notes, "no runnable
  utility tasks"])` (mirrors `TSTRSuite`).
- Aggregate:
  - `all_ratios` = every `(task, model)` ratio. `score = clamp(mean(all_ratios), 0, 1)`.
  - For each model name: `per_model_mean[name] = mean(ratios for that model across tasks)`.
  - `ratio_spread = std(list(per_model_mean.values()))` (population std; 0.0 if <2 models — there
    are always 3, so always defined).
- scalars:
  - `"score"`: the mean ratio (headline, [0,1]).
  - `"utility_ratio_mean"`: same value (explicit label).
  - `"ratio_linear"`, `"ratio_random_forest"`, `"ratio_hist_gbdt"`: per-model mean ratios.
  - `"ratio_spread"`: std across the three per-model means.
- `MetricResult(spec=self.spec, scalars=scalars, per_column=per_column, notes=notes)`.

## metric_info wiring (`src/syneva/core/metric_info.py`)

- `METRIC_NAMES["model_panel_utility"] = "Model-panel utility (TSTR across model families)"`.
- `METRIC_INFO["model_panel_utility"]`: neutral description, no direction claim, e.g. *"Trains linear,
  random-forest, and gradient-boosting models on the synthetic data and on the real data, then compares
  how well each performs on a held-out real test set. Reports the average ratio and the per-family
  breakdown."*
- `RAW_HINT["model_panel_utility"]`: the raw measurements are utility ratios where higher is better
  (TSTR approaching TRTR). Follow the existing RAW_HINT convention/shape for a "higher is better" ratio.
- `_SCALAR_LABELS` entries: `"utility_ratio_mean": "Mean utility ratio"`, `"ratio_linear": "Utility
  ratio — linear model"`, `"ratio_random_forest": "Utility ratio — random forest"`, `"ratio_hist_gbdt":
  "Utility ratio — gradient boosting"`, `"ratio_spread": "Spread across model families (std)"`.

(Guard tests in `tests/unit/core/test_metric_info.py` enforce that every registered metric has a
METRIC_NAMES + METRIC_INFO entry and every scalar key has a label; these entries satisfy them.)

## SynthEval parity manifest

Inspect `tests/parity/test_syntheval_parity.py`. If SynthEval credits a multi-model / multiple-classifier
utility capability, mark the corresponding manifest entry implemented (pointing at `model_panel_utility`);
if there is no such entry, leave the manifest unchanged and note it.

## Registration / import

The metric auto-registers via `@registry.register`. Ensure the new module is imported so registration
runs — follow how the OTHER extended utility metric (`src/syneva/utility/extended/multi_target.py`) is
imported/discovered (e.g. an `__init__.py` import or the package's metric-discovery mechanism). Match
that exact pattern so `model_panel_utility` appears in the registry.

## Testing

- **Panel runs:** with a real classification task, `model_panel_utility` produces `score`,
  `utility_ratio_mean`, `ratio_linear`, `ratio_random_forest`, `ratio_hist_gbdt`, `ratio_spread`, all
  finite and `score ∈ [0,1]`; `per_column` has a `<task>/<model>` entry for each of the 3 models.
- **Identical data:** real == synthetic ⇒ each per-model ratio ≈ 1 (within tolerance) and
  `ratio_spread` small.
- **Holdout forwarded:** constructing with `holdout=` runs and tests on the holdout (assert it runs and
  score sensible).
- **Behavior preservation:** existing `tstr_suite` / `multi_target_utility` tests pass UNCHANGED; the
  golden snapshot stays green (NEVER `--update-golden`). `build_pipeline`'s single-RF output is
  unchanged (if `_build_preprocessor` is extracted, an existing TSTR test passing proves it).
- **Registry/runner integration:** `evaluate(real, syn, meta, tiers=("core","extended"),
  run_utility=True, utility_tasks=[...])` includes a `model_panel_utility` result with no error; and
  `evaluate(..., preset="full", run_utility-implied)` includes it.
- **metric_info guards:** `tests/unit/core/test_metric_info.py` passes with the new entries.
- 85% coverage gate holds.

## Open questions
None outstanding. Regression panel uses Ridge/RF/HistGBDT mirroring the classification trio; the
headline `score` is the mean utility ratio across all (task, model) pairs, clamped to [0,1].
