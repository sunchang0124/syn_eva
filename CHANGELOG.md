# Changelog

All notable changes follow [keep-a-changelog](https://keepachangelog.com).

## [Unreleased]

### Added
- Streamlit web UI (`syn-eva ui`, optional `[ui]` extra): upload real/synthetic
  data, edit column metadata, select evaluators, and view/download the scorecard.

### Changed
- Column metadata has a new `quasi_identifier` role, separate from `sensitive`
  (#34). `k_anonymity` now groups on `quasi_identifier` columns instead of
  `sensitive` ones, and `attribute_disclosure` infers the `sensitive` columns
  from the declared quasi-identifiers (falling back to every other usable
  column when none are declared). Existing metadata must add
  `"quasi_identifier": true` to the linking columns to keep k-anonymity results;
  without it, `k_anonymity` is skipped.

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
