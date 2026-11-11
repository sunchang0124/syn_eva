# Changelog

All notable changes follow [keep-a-changelog](https://keepachangelog.com).

## [Unreleased]

### Added
- Streamlit web UI (`syn-eva ui`, optional `[ui]` extra): upload real/synthetic
  data, edit column metadata, select evaluators, and view/download the scorecard.
- `syn-eva evaluate` and `syn-eva benchmark` take `--random-state`, and the
  Streamlit sidebar has a "Random seed" input in both modes (#93). Both default
  to 42, so runs that don't set them are unchanged.

### Changed
- Column metadata has a new `quasi_identifier` role, separate from `sensitive`
  (#34). `k_anonymity` now groups on `quasi_identifier` columns instead of
  `sensitive` ones, and `attribute_disclosure` infers the `sensitive` columns
  from the declared quasi-identifiers (falling back to every other usable
  column when none are declared). Existing metadata must add
  `"quasi_identifier": true` to the linking columns to keep k-anonymity results;
  without it, `k_anonymity` is skipped.

### Fixed
- `Metadata.infer`, and so `evaluate` without explicit metadata, no longer
  crashes on column labels that aren't strings, such as the `0, 1, ...` of a
  DataFrame built from a NumPy array or read with `header=None` (#35). The
  utility metrics (`tstr_suite`, `model_panel_utility`, `multi_target_utility`,
  `feature_importance_spearman`) also work on such frames: their models read
  integer labels as column positions, which picked the wrong columns or failed,
  and rejected a mix of string and non-string labels.
- `evaluate(random_state=...)` now controls every random component of a run
  (#42). `c2st`, `discriminative_score`, `sliced_wasserstein`, `mmd`,
  `hitting_rate`, `pmse`, `pca_scatter` and the shared Gower subsampling used a
  hardcoded seed of 42, so a seed sweep repeated the same splits, subsamples and
  projections and `RunInfo.random_state` reported a seed that was never used.
  Each of these, and every metric that builds distances (`dcr`, `nndr`,
  `epsilon_identifiability`, `alpha_precision_beta_recall`, `authenticity`,
  `nn_adversarial_accuracy`), now takes `random_state` (default 42), so results
  at the default seed are unchanged. `mia_auc` and `attribute_disclosure` take
  it too, though their euclidean distances never sample rows.
  `minority_utility_gap` now also passes its seed to its classifier.

## [0.1.0rc1] - 2026-06-12

### Added
- Extended-tier metrics: sliced Wasserstein, Jensen-Shannon, C2ST (congruence);
  alpha-precision/beta-recall, authenticity, PCA scatter (coverage); MIA AUC,
  DP ledger pass-through (compliance); multi-target, feature-importance Spearman,
  discriminative score (utility).
- PDF renderer via weasyprint (optional `syneva[pdf]` extra).
- `syn-eva` CLI with an `evaluate` subcommand (JSON/HTML/PDF, `--cs` filter).
- Property-based invariant tests and a golden JSON snapshot.
- README quickstart, per-metric documentation stubs, and a runnable example.

### Internals
- 85% coverage gate; pytest uses importlib import mode.

## [0.1.0a2] - 2026-06-12

### Added
- Full core scorecard: 16 core metrics across Congruence/Coverage/Compliance/Utility.
- HTML renderer with embedded matplotlib plots.
- Auto-registration of core metrics on `import syneva`.

### Internals
- `evaluate()` accepts `utility_tasks` and `run_utility`; defaults skip utility unless requested.

## [0.1.0a0]

### Added
- Initial scaffolding for v0.1.
