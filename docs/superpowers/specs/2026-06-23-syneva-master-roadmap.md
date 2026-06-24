# syneva — Master Roadmap (the full picture)

**Date:** 2026-06-23
**Status:** Living roadmap. Each component below gets its own design spec + implementation plan.
**Purpose:** One place that shows the entire target system — every component, how they connect, the
build order, and how each contributes to the publication — so component specs are written against a
shared architecture rather than in isolation.

---

## 1. Thesis

Make syneva a **strict superset of [SynthEval](https://github.com/schneiderkamplab/syntheval)**
(arXiv 2404.15821; Springer DMKD) and go beyond it, targeting an **ML / data-mining venue**.

- **Parity:** cover 100% of SynthEval's metrics, its benchmark/ranking, Gower distance, holdout, and
  presets.
- **Superset moat (already built):** the published **7 Cs** framework; richer Coverage/diversity
  metrics; a **Streamlit GUI**; **HTML/PDF reports**; **normalized + actual** dual reporting with
  verdict badges and correct per-metric direction hints.
- **Headline novelties (all four, per decision 2026-06-23):** minority/outlier preservation,
  longitudinal evaluation, causal-fidelity evaluation, and a validated 7 Cs aggregate.

Risk noted and accepted: four headline novelties in one paper is ambitious; each is built as an
independent, shippable phase so the tool is sound regardless of final paper framing.

---

## 2. Target architecture

Everything plugs into syneva's existing spine; no rewrite. New work is either (a) new metric classes in
the registry, (b) new cross-cutting capabilities threaded through `evaluate`, or (c) new top-level
subsystems (benchmark, longitudinal backend).

```
                         syneva.evaluate() / evaluate_with()
                                      │
   ┌──────────────┬──────────────┬───┴───────────┬───────────────┬─────────────┐
   │ Metadata     │ Distance     │ Holdout       │ Config objects │ MetricRegistry│
   │ (+subgroup,  │ backends:    │ (real test    │ UtilityTask    │  select(tiers,│
   │  protected,  │ euclidean /  │  set)         │ FairnessSpec   │  cs, preset)  │
   │  temporal)   │ gower        │               │ SubgroupSpec   │               │
   └──────────────┴──────────────┴───────────────┴───────────────┴───────┬───────┘
                                                                          │
        ┌──────────────────────────── Metric classes ────────────────────┘
        │  Dimensions (spec.c): congruence · coverage · compliance · utility
        │                       · fairness (new) · preservation (new) · temporal (new)
        ▼
   MetricResult ──► Report (by_c, aggregated, to_json/html/pdf, metric_info)
        │
        ├──► Renderer + Streamlit UI  (normalized / actual modes, verdicts)
        └──► Benchmark engine (N synthetic datasets ─► ranking ─► leaderboard report)
```

**Cross-cutting capabilities (threaded through `evaluate`/`evaluate_with`):**
- **Distance backends** — `distance="euclidean"|"gower"`; a pluggable encoder in
  `compliance/_encode.py` (add a Gower similarity/distance matrix). Used by all NN-based metrics
  (DCR, NNDR, NNAA, ε-identifiability, attribute disclosure, authenticity, alpha/beta).
- **Holdout** — optional `holdout: DataFrame` (a real test set). Utility trains on real/synthetic and
  tests on holdout; privacy uses holdout rows as non-members; benchmark uses (train, holdout).
- **Config objects** — `UtilityTask` (exists), `FairnessSpec` (protected attribute + outcome),
  `SubgroupSpec` (which groups/columns define minorities), `TemporalSpec` (entity id + time column +
  sequence layout). Each is optional and only activates its metrics.
- **Presets** — named selections (`full`, `fast`, `privacy`, `core`) mapping to (tiers, cs, flags),
  plus custom JSON config (SynthEval-compatible where sensible).

**New dimensions (new `spec.c` values, extend the `C` Literal + `C_INFO`):**
- `fairness` — bias preservation/introduction.
- `preservation` — minority/outlier/tail/subgroup retention (headline novelty #1).
- `temporal` — longitudinal/sequential fidelity (headline novelty #2).

**New subsystems:**
- `src/syneva/benchmark/` — `benchmark(synthetics, real, holdout=None, ...) -> BenchmarkReport`;
  ranking strategies (`linear`, `normal`, `quantile`) in `benchmark/ranking.py`; leaderboard rendering.
- `src/syneva/backends/longitudinal.py` — currently a placeholder; becomes a real backend with a
  sequence data model + temporal metrics.
- `src/syneva/causal/` — causal-fidelity metrics (ATE/CATE recovery, causal-graph distance).
- Aggregate calibration lives in `core` (a validated weighting for `Report.aggregated`) plus an
  `experiments/` study.

---

## 3. Full metric inventory (target)

Legend: ✅ exists · 🅿️ parity (SynthEval) · 🆕 novelty.

**Congruence** — distribution & relationship fidelity
✅ ks_statistic, tvd, wasserstein, correlation_difference, pmse, sliced_wasserstein, jsd, c2st
🅿️ dimension_wise_means, ci_overlap, hellinger, quantile_mse, mutual_information_difference, mmd

**Coverage** — variety, novelty, generalization
✅ category_coverage, range_coverage, novelty_rate, entropy_ratio, alpha_precision_beta_recall, authenticity, pca_scatter
🅿️ nn_adversarial_accuracy

**Compliance** — privacy & disclosure
✅ dcr, nndr, k_anonymity, identical_match_rate, mia_auc, dp_ledger
🅿️ hitting_rate, epsilon_identifiability, attribute_disclosure

**Utility** — usefulness for modeling
✅ tstr_suite, multi_target_utility, feature_importance_spearman, discriminative_score
🅿️ (enhancement) multi-classifier TSTR/TRTR + explicit AUROC difference + holdout

**Fairness** 🆕 (new dimension)
🅿️ statistical_parity · 🆕 (optional) equalized_odds_difference, disparate_impact

**Preservation** 🆕 (new dimension — headline novelty #1)
🆕 rare_category_retention, tail_coverage, minority_class_density, subgroup_fidelity,
   minority_utility_gap, minority_privacy_risk (final set fixed in its own spec)

**Temporal** 🆕 (new dimension — headline novelty #2, longitudinal)
🆕 trajectory_distribution, transition_matrix_difference, temporal_autocorrelation,
   event_timing, sequence_level_dcr (final set fixed in its own spec)

**Causal** 🆕 (headline novelty #3)
🆕 ate_recovery, cate_recovery, causal_graph_distance (final set fixed in its own spec)

---

## 4. Components & their specs (build units)

Each row is a future design spec → implementation plan. "Touches" lists the main files.

| # | Component | Adds | Depends on | Touches | Status |
|---|---|---|---|---|---|
| C1 | **Parity 0a — fidelity metrics** | 7 metrics (table above) + parity manifest | none | congruence/, coverage/extended/, metric_info, tests/parity | ✅ done |
| C2 | **Parity 0b — privacy + fairness** | hitting_rate, epsilon_identifiability, attribute_disclosure; `fairness` dim + statistical_parity; `FairnessSpec` | C1; config plumbing | compliance/, new fairness/, runner, metadata | ✅ done |
| C3 | **Parity 0c — methodology rigor** | Gower distance backend (0c-a); holdout support (0c-b); presets (0c-c); multi-classifier utility (0c-d) | C1–C2 | _encode, runner, utility/, cli, ui | ✅ done |
| C4 | **Benchmark & ranking engine** | rank N synthetic datasets; linear/normal/quantile; leaderboard report + UI tab | C3 (presets/holdout) | new benchmark/, render/, cli, ui | ⏭️ next |
| C5 | **Preservation (novelty #1)** | `preservation` dim + minority/outlier/tail/subgroup metrics; `SubgroupSpec` | C1; benchmark for experiments | new preservation/, runner, metadata, ui |
| C6 | **Validated aggregate (novelty #4)** | calibrated 7 Cs score predicting downstream utility; weighting in Report | C4 (benchmark) + many datasets | core/report, experiments/ |
| C7 | **Longitudinal (novelty #2)** | sequence data model + `temporal` dim + temporal metrics | C1; backend | backends/longitudinal, new temporal/, metadata |
| C8 | **Causal (novelty #3)** | `causal` dim + ATE/CATE/graph-distance metrics; needs treatment/outcome config | C1 | new causal/, runner, metadata |
| C9 | **Paper experiments + write-up** | reproducible benchmark vs SynthEval; ablations; novelty studies; manuscript | C4–C8 | experiments/, paper/ |

---

## 5. Build order & dependency rationale

```
C1 (0a) ─► C2 (0b) ─► C3 (0c) ─► C4 (benchmark) ─┬─► C5 (preservation) ─┐
                                                 ├─► C7 (longitudinal)  ├─► C9 (paper)
                                                 ├─► C8 (causal)        │
                                                 └─► C6 (validated aggregate)
```

- **C1→C2→C3 first** because they complete SynthEval parity and build the shared plumbing (config
  objects, distance backends, holdout, presets) every later component reuses.
- **C4 (benchmark) next** because every empirical claim and novelty study runs through multi-dataset
  evaluation + ranking.
- **C5–C8 (novelties)** are independent of each other and can be built in any order / in parallel once
  C4 exists; recommended order C5 (most tractable) → C6 → C7 → C8.
- **C9 (paper)** consumes the rest.

This sequence keeps every step shippable: after C1 syneva already beats SynthEval on fidelity coverage;
after C4 it is a full superset; each novelty then extends the lead.

---

## 6. Publication plan (what each component contributes)

- **Claim 1 — Superset:** the parity manifest (C1–C3) + benchmark (C4) show syneva computes every
  SynthEval metric and ranks datasets at least as well, on the same data, reproducibly.
- **Claim 2 — Novel axes:** preservation (C5), longitudinal (C7), causal (C8) measure properties no
  existing tabular tool reports; experiments show generators that look good on global fidelity yet fail
  these axes (the compelling narrative).
- **Claim 3 — A usable, validated framework:** the validated aggregate (C6) gives a single score shown
  to predict downstream utility, and the GUI + dual reporting make the framework usable — addressing
  SynthEval's "no single score" and CLI-only limitations.
- **Experiments (C9):** multiple public tabular datasets × multiple generators (e.g. CTGAN, TVAE,
  diffusion, copulas) × syneva vs SynthEval; ablations on distance backend and aggregate weighting;
  novelty case studies (rare-subgroup destruction; a longitudinal dataset; a causal benchmark).
- **Baselines:** SynthEval (primary), plus SDMetrics/SDV reports where relevant.

---

## 7. How we proceed

For each component C1…C9, in build order: brainstorm → design spec in
`docs/superpowers/specs/` → implementation plan in `docs/superpowers/plans/` → build (TDD,
subagent-driven) → verify. This master roadmap is updated as decisions are made.

- **C1 (0a)** spec already written: `2026-06-23-syneva-syntheval-parity-0a-fidelity.md`.
- Next actions are sequential; we only open the next component's spec once the current one is planned.

## Open questions (to resolve as each component is reached)
- C2: is `fairness` a new top-level dimension or folded under compliance? (Leaning: new dimension.)
- C5: is minority/outlier preservation its own `preservation` dimension or an extension of Coverage?
- C6: aggregate weighting — learned vs fixed; what downstream-utility target defines "predictive"?
- C7: longitudinal data model (long vs wide; fixed vs variable sequence length) and reference datasets.
- C9: target venue shortlist and the exact dataset/generator matrix.
