# Changelog

All notable changes follow [keep-a-changelog](https://keepachangelog.com).

## [Unreleased]

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
