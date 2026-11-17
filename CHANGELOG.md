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
- `evaluate` now validates the holdout against the column metadata, as it
  already did for the real and synthetic data (#41). A holdout whose column
  types don't match, such as a numeric `age` read as text because it contains
  `'unknown'`, used to be coerced silently, which changed the holdout-based
  `dcr`, `nndr` and `mia_auc` scores with no warning. It now raises
  `MetadataError` with a message starting `holdout:`. Correctly typed holdouts
  give the same results as before. The type-mismatch message now names the
  types the same way on every Python version (`declared numeric but is
  categorical`); on Python 3.11 and later it used to read `declared
  ColumnType.NUMERIC but is ColumnType.CATEGORICAL`.
- `statistical_parity` now gives a fairness spec the maximum `parity_drift` of
  1.0, with a note, when a protected group from the real data is missing from
  the synthetic data (#40). It used to compare only the groups present in each
  frame, so erasing a group scored as a smaller gap: 0.7 for a two-group spec,
  and up to 1.0 when the remaining groups happened to keep the real gap. Groups
  are now taken from the real data, and groups that appear only in the
  synthetic data are ignored. Scores where every real group is present are
  unchanged.
- `quantile_mse` and `dimension_wise_means` now skip a numeric column that is
  constant in the real data, with a note (#39). Both standardise by the real
  standard deviation, so such a column used to be divided by `1e-12`, and any
  synthetic difference, even float noise, brought the score for all columns to
  0. Scores for data without constant real columns are unchanged up to rounding
  (the `1e-12` term is gone). If every numeric column is constant, the metric
  is skipped and the reason names them.
- `correlation_difference` now compares every column pair on the same 0-to-1
  scale (#38). Numeric–categorical pairs use the correlation ratio η instead of
  η², and each Pearson difference is capped at 1, so a reversed relationship is
  still penalised but no pair contributes more than 1, the most a categorical
  pair can. Scores for this metric change: η² put numeric–categorical pairs on a
  squared scale (shrinking a difference when η₁ + η₂ < 1 and enlarging it when
  η₁ + η₂ > 1), and a Pearson pair that flipped sign could count up to twice as
  much as any other pair; when such pairs made up most of the scored pairs, the
  score was clipped to 0.
- The UI's subgroup builder now chooses between a min/max range and a value list
  from the column types in the metadata editor, not the pandas dtype (#36). A
  bool column, or a number column re-typed as categorical, used to get a range
  that the subgroup metrics rejected, so the subgroup was never evaluated. A
  text column re-typed as numeric gets its bounds from the numbers in it, and one
  with no numbers at all is flagged in the sidebar; while it is picked, no
  subgroup is built, rather than one without it that would cover more rows.
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
