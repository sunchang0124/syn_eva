# syneva — SynthEval parity, Phase 0a: fidelity metrics — Design

**Date:** 2026-06-23
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 public API (`registry`, `MetricSpec`, `MetricResult`, `Metadata`, `ColumnType`, `encode_pair`, the metric-info registry `describe_metric`/`display_name`/`raw_hint`/`humanize_scalar`).

## Background

We are making syneva a strict superset of [SynthEval](https://github.com/schneiderkamplab/syntheval)
([arXiv 2404.15821](https://arxiv.org/abs/2404.15821)) and then going beyond it. SynthEval offers
~14 utility/fidelity metrics, 6 privacy metrics, 1 fairness metric, a multi-dataset benchmark/ranking
module, Gower-distance handling, holdout support, and config presets. syneva already covers many of
its fidelity metrics (correlation difference, KS, TVD, pMSE, feature-importance correlation, TSTR/TRTR)
and adds a richer Coverage suite, a GUI, and dual normalized/raw reporting.

Phase 0 closes the remaining parity gap. It is split into three specs:

- **0a (this spec):** the fidelity/coverage metrics SynthEval has that syneva lacks. No new
  infrastructure — pure registry additions.
- **0b (next):** privacy metrics (hitting rate, epsilon identifiability, attribute disclosure) and a
  new **Fairness** dimension (statistical parity), which need protected/sensitive-column config.
- **0c (later):** methodology rigor — Gower-distance option, holdout test-set support, named presets,
  multi-classifier utility.

## Goal

Add seven metrics so that syneva's **Congruence** and **Coverage** dimensions fully cover SynthEval's
fidelity surface, each with a normalized 0–1 score, a correct raw-direction hint, and a full-name +
plain-language description. Add a machine-checked **parity manifest** that fails CI if any SynthEval
metric we claim to cover is not actually registered.

## Scope

**In scope (7 new metrics):**

| name | C / tier | scope | requires_real |
|---|---|---|---|
| `dimension_wise_means` | congruence / core | per-column | yes |
| `ci_overlap` | congruence / core | per-column | yes |
| `hellinger` | congruence / core | per-column | yes |
| `quantile_mse` | congruence / core | per-column | yes |
| `mutual_information_difference` | congruence / extended | table-level | yes |
| `mmd` | congruence / extended | table-level | yes |
| `nn_adversarial_accuracy` | coverage / extended | table-level | yes |

Plus: the parity-manifest module + test.

**Out of scope (deferred):** privacy metrics, fairness dimension, Gower distance, holdout, presets,
multi-classifier utility (0b/0c). No UI changes are needed — the existing renderer and Streamlit app
pick up new registered metrics automatically.

## Architecture / placement

Each metric is a self-contained class following the existing pattern (a `@registry.register` class with
a `ClassVar[MetricSpec]` and a `compute(self, real, synthetic, meta) -> MetricResult`). Files:

- Create `src/syneva/congruence/dimension_wise_means.py`, `ci_overlap.py`, `hellinger.py`, `quantile_mse.py`
- Create `src/syneva/congruence/extended/mutual_information_difference.py`, `mmd.py`
- Create `src/syneva/coverage/extended/nn_adversarial_accuracy.py`
- Register via the existing side-effect imports in `congruence/__init__.py`,
  `congruence/extended/__init__.py`, `coverage/extended/__init__.py`.
- Table-level numeric encoding reuses `syneva.compliance._encode.encode_pair` (shared real+synthetic space).

Every metric returns `scalars={"score": <0-1>, <raw_key>: <value>, ...}` and (for per-column metrics)
a `per_column` dict. The headline `score` is normalized so 1.0 = ideal; the raw key(s) carry the
objective statistic for the "actual measured values" view.

## Per-metric design

Notation: numeric columns are those with `ColumnType.NUMERIC`; categorical with `ColumnType.CATEGORICAL`.
`eps = 1e-12` guards division. Scores clamp to `[0, 1]`.

### 1. `dimension_wise_means` (Congruence, core, per-column)
- **Measures:** whether each numeric column's mean is preserved.
- **Compute:** per numeric column, `d = |mean_real - mean_syn| / (std_real + eps)` (standardized mean gap).
- **score:** `mean over columns of clamp(1 - d)`.
- **scalars:** `{score, mean_abs_std_diff}`; `per_column[col] = {"std_mean_diff": d}`.
- **raw direction:** standardized mean gap; 0 = identical, lower is better.

### 2. `ci_overlap` (Congruence, core, per-column)
- **Measures:** statistical agreement of column means via 95% confidence-interval overlap.
- **Compute:** per numeric column, CI = `mean ± 1.96 * std/sqrt(n)` for real and syn. Overlap fraction =
  `overlap_width / mean(CI_width_real, CI_width_syn)`, clamped `[0, 1]` (0 if disjoint).
- **score:** mean overlap fraction over columns.
- **scalars:** `{score, mean_ci_overlap}`; `per_column[col] = {"ci_overlap": frac}`.
- **raw direction:** overlap fraction in `[0, 1]`; higher is better.

### 3. `hellinger` (Congruence, core, per-column)
- **Measures:** per-column distributional distance for both numeric and categorical columns.
- **Compute:** categorical — Hellinger over category-frequency vectors aligned on the union of categories:
  `H = sqrt(0.5 * sum((sqrt(p_i) - sqrt(q_i))**2))`. Numeric — bin real and syn into a shared set of bins
  (union range, `min(20, ...)` bins), normalize counts to probabilities, then the same formula.
- **score:** `1 - mean H` over columns.
- **scalars:** `{score, mean_hellinger}`; `per_column[col] = {"hellinger": H}`.
- **raw direction:** Hellinger in `[0, 1]`; 0 = identical, lower is better.

### 4. `quantile_mse` (Congruence, core, per-column numeric) — tail accuracy
- **Measures:** agreement of the full quantile function, emphasizing tails.
- **Compute:** per numeric column, take quantiles at `q in {0.05, 0.1, ..., 0.95}`; standardize by
  `std_real + eps`; `qmse = mean((q_real - q_syn)/scale)**2`.
- **score:** `mean over columns of clamp(1 - sqrt(qmse))`.
- **scalars:** `{score, mean_quantile_mse}`; `per_column[col] = {"quantile_mse": qmse}`.
- **raw direction:** standardized quantile MSE; 0 = identical, lower is better.

### 5. `mutual_information_difference` (Congruence, extended, table-level)
- **Measures:** whether pairwise dependency structure (beyond linear correlation) is preserved.
- **Compute:** discretize numerics into 10 quantile bins, keep categoricals as labels; for every column
  pair compute normalized mutual information `NMI = MI / sqrt(H_i * H_j)` (0 if either entropy is 0) using
  `sklearn.metrics.mutual_info_score`. Build the NMI matrix for real and for syn; `raw = mean absolute
  difference over the upper triangle`.
- **score:** `clamp(1 - raw)`.
- **scalars:** `{score, mean_mi_diff}`.
- **raw direction:** mean |ΔNMI|; 0 = identical structure, lower is better.

### 6. `mmd` (Congruence, extended, table-level)
- **Measures:** overall multivariate distributional discrepancy via a kernel two-sample statistic.
- **Compute:** `X_real, X_syn = encode_pair(real, synthetic, meta)`. Cap each at `n=500` rows
  (seeded subsample via `numpy.random.default_rng(42)`) for cost. RBF kernel with bandwidth =
  median pairwise distance (median heuristic). `mmd2 = mean(K_xx) + mean(K_yy) - 2*mean(K_xy)`;
  `mmd = sqrt(max(0, mmd2))`.
- **score:** `clamp(1 - mmd)`.
- **scalars:** `{score, mmd}`.
- **raw direction:** MMD ≥ 0; 0 = identical, lower is better.

### 7. `nn_adversarial_accuracy` (Coverage, extended, table-level)
- **Measures:** whether synthetic points are over-close (memorization) or too far (poor coverage), via
  nearest-neighbour separability — the NN counterpart to the classifier-based `discriminative_score`.
- **Compute:** `X_real, X_syn = encode_pair(...)`. Adversarial accuracy following Yale et al.:
  `AA = 0.5 * ( mean[ d(x_real, NN_syn) > d(x_real, NN_real_excl_self) ] + mean[ d(x_syn, NN_real) >
  d(x_syn, NN_syn_excl_self) ] )` using 1-NN distances. AA ≈ 0.5 means real and synthetic are
  indistinguishable (ideal); AA → 1 means too separable; AA → 0 means synthetic sits on top of real
  (memorization).
- **score:** `clamp(1 - 2*|AA - 0.5|)`.
- **scalars:** `{score, nn_adversarial_accuracy}`.
- **raw direction:** AA in `[0, 1]`; **0.5 is ideal** (closer to 0.5 is better).

## Metric-info entries (required)

For every new metric, add entries to `src/syneva/core/metric_info.py`:
`METRIC_NAMES` (full name), `METRIC_INFO` (neutral "what it measures", no direction claim),
`RAW_HINT` (scale + correct direction per the tables above), and `_SCALAR_LABELS` for any new scalar
keys (`mean_abs_std_diff`, `mean_ci_overlap`, `mean_hellinger`, `mean_quantile_mse`, `mean_mi_diff`,
`mmd`, `nn_adversarial_accuracy`). The existing guard tests
(`test_every_registered_metric_has_a_full_name`, `..._description`, `..._raw_hint`,
`test_base_description_makes_no_direction_claim`) enforce this — the suite fails if any entry is missing
or sneaks a direction claim into the base description.

Full names (for `METRIC_NAMES`): "Dimension-wise means", "Confidence-interval overlap",
"Hellinger distance", "Quantile MSE", "Mutual-information difference", "Maximum mean discrepancy",
"Nearest-neighbour adversarial accuracy".

## Parity manifest

Create `tests/parity/syntheval_manifest.py` with a dict mapping each SynthEval metric code to its syneva
status:

```python
SYNTHEVAL_PARITY = {
    # code -> ("implemented", syneva_metric_name) | ("planned", phase) | ("na", reason)
    "corr_diff": ("implemented", "correlation_difference"),
    "ks_test": ("implemented", "ks_statistic"),     # KS + TVD pair
    "p_MSE": ("implemented", "pmse"),
    "fio": ("implemented", "feature_importance_spearman"),
    "cls_acc": ("implemented", "tstr_suite"),
    "auroc_diff": ("implemented", "tstr_suite"),
    "dwm": ("implemented", "dimension_wise_means"),
    "cio": ("implemented", "ci_overlap"),
    "h_dist": ("implemented", "hellinger"),
    "q_mse": ("implemented", "quantile_mse"),
    "mi_diff": ("implemented", "mutual_information_difference"),
    "mmd": ("implemented", "mmd"),
    "nnaa": ("implemented", "nn_adversarial_accuracy"),
    "pca": ("implemented", "pca_scatter"),
    "nndr": ("implemented", "nndr"),
    "dcr": ("implemented", "dcr"),
    "mia": ("implemented", "mia_auc"),
    "hit_rate": ("planned", "0b"),
    "eps_risk": ("planned", "0b"),
    "att_discl": ("planned", "0b"),
    "statistical_parity": ("planned", "0b"),
}
```

Test (`tests/parity/test_syntheval_parity.py`): for every entry marked `"implemented"`, assert the named
metric is present in the global `registry`. This makes "superset" a CI-enforced invariant and gives the
paper a reproducible coverage claim. As 0b/0c land, entries flip from `planned` to `implemented`.

## Testing

TDD per metric (`tests/unit/congruence/...` and `tests/unit/coverage/extended/...`):
- **Identical-data invariant:** `metric.compute(df, df, meta)` → `score >= ~0.95` (or for `nn_adversarial_accuracy`, AA near 0.5 → score high), raw value near its ideal.
- **Divergence invariant:** a clearly shifted/different synthetic → lower score and a worse raw value in
  the correct direction (e.g. larger Hellinger, larger MMD, AA away from 0.5).
- Use small constructed DataFrames where exact behaviour matters, plus the `adult_income` fixtures for a
  realistic check.

Integration: extend `tests/integration/test_full_core_scorecard.py` (or add a focused test) asserting the
new metrics appear in a full `evaluate(..., tiers=("core","extended"))` run. The coverage gate (85%) must
hold; new metrics are normal source files (not excluded).

## Out of scope / future (recorded so the manifest stays honest)
- Privacy: `hit_rate`, `eps_risk`, `att_discl` → spec 0b.
- Fairness dimension + `statistical_parity` → spec 0b.
- Gower distance, holdout support, presets, multi-classifier utility → spec 0c.
- Benchmark/ranking subsystem, minority/outlier-preservation, longitudinal, causal, validated aggregate →
  later phases / the paper's headline contributions.

## Open questions
None outstanding for 0a.
