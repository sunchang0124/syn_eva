# syneva — Phase 0c-b: holdout test-set support — Design

**Date:** 2026-06-23
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + 0a + 0b + 0c-a (the `Neighbors` helper, `encode_pair`). Public API:
`evaluate`/`evaluate_with`, `_instantiate`, `Metadata`, `SynevaError`.
**Roadmap:** component C3 (methodology rigor), sub-spec C3b. Siblings: presets, multi-classifier utility.

## Goal

Add an optional `holdout` real test set to `evaluate()` so utility metrics test on truly unseen data,
MIA uses it as the non-member set, and the DCR/NNDR privacy metrics gain a held-out baseline. Holdout is
opt-in: with `holdout=None` every metric behaves exactly as today (behavior-preserving).

## Scope

**In scope:** `holdout` param threaded through the runner; an `encode_frames` helper for consistent
3-frame encoding; a `holdout` extension to `Neighbors`; holdout branches in five metrics
(`tstr_suite`, `multi_target_utility`, `mia_auc`, `dcr`, `nndr`); schema validation; a UI uploader.

**Out of scope (separate specs):** presets, multi-classifier utility. Holdout is not consumed by
congruence/coverage/fairness metrics.

## Threading

- `evaluate()` / `evaluate_with()` gain `holdout: pd.DataFrame | None = None` (after `distance`).
- Validation in `evaluate_with`: if `holdout is not None`, its columns must match `synthetic`'s
  (same set); mismatch → `SynevaError`.
- Generalize `_instantiate` to pass `holdout=` to any metric whose constructor accepts it (alongside
  `tasks`/`specs`/`distance`/`random_state`). The five metrics gain a `holdout: pd.DataFrame | None = None`
  constructor field; all others ignore it. The `Metric.compute` protocol is unchanged.

## Consistent encoding (`encode_frames`)

The privacy baseline and MIA compare distances across three frames (real, synthetic, holdout), so all
three must live in **one** encoding space. Add to `src/syneva/compliance/_encode.py`:

```
encode_frames(frames: list[pd.DataFrame], meta: Metadata) -> list[np.ndarray]
```

It concatenates all frames, fits one `StandardScaler` (numeric) and one-hot (categorical, union of
categories) on the union, and splits back into per-frame matrices — the n-frame generalization of
`encode_pair`. `encode_pair` is left untouched (hot path; behavior must not change); `encode_frames`
is the new, separate function. (A future cleanup may have `encode_pair` delegate to it, out of scope here.)

## `Neighbors` holdout extension

`Neighbors.__init__` gains an optional `holdout: pd.DataFrame | None = None`. When provided:
- **euclidean:** encode all present frames together via `encode_frames([real, synthetic, holdout])` so
  `X_real`, `X_syn`, `X_holdout` share one space. (When `holdout is None`, it still uses the 2-frame
  `encode_pair` — euclidean behavior for every existing call is unchanged.)
- **gower:** also precompute `gower_matrix(holdout, real, meta)` (capped consistently).
- New methods `holdout_to_real(k=1)` (per holdout row, distance to its k-th nearest real record) and
  `holdout_self(k=1)` (per holdout row, distance to its k-th nearest *other* holdout record). Both raise
  if no holdout was given. They mirror `syn_to_real` / `syn_self`.

This keeps `Neighbors` the single distance authority. DCR/NNDR call `nb.holdout_to_real(1)` for the
baseline; everything else is as in 0c-a.

## Metric holdout semantics (all conditional; `holdout=None` ⇒ unchanged)

### Utility — `tstr_suite`
With holdout: for each task, train the model on the **full real** data (TRTR) and on the **full
synthetic** data (TSTR), and **evaluate both on the holdout** (`X_test = holdout[features]`,
`y_test = holdout[target]`). Without holdout: today's internal `train_test_split(real)`. The utility
ratio = TSTR / TRTR as today.

### Utility — `multi_target_utility`
Gains a `holdout` field and forwards it to the inner `TSTRSuite(tasks=..., random_state=..., holdout=...)`.

### Privacy — `mia_auc`
With holdout: encode `[real, synthetic, holdout]` together (`encode_frames`); fit nearest-neighbour on
`X_syn`; **members = real** (label 1), **non-members = holdout** (label 0); the membership score is the
negative nearest-synthetic distance; `mia_auc = roc_auc_score(labels, scores)`; `score = 1 − max(0,
auc − 0.5) * 2` (unchanged mapping). Without holdout: today's internal member/non-member split.

### Privacy baseline — `dcr`
With holdout: `d_syn = nb.syn_to_real(1)`, `d_hold = nb.holdout_to_real(1)` (same encoding).
`p05_syn = quantile(d_syn, 0.05)`, `p05_hold = quantile(d_hold, 0.05)`. **Score becomes relative:**
`score = clamp(p05_syn / (p05_hold + eps))` — 1.0 means synthetic sits no closer to the training reals
than a genuine held-out real sample (private); `< 1` means synthetic is suspiciously closer
(memorization risk). Extra scalars `median_dcr_holdout`, `p05_dcr_holdout`. Without holdout: today's
absolute `score = min(1, p05_syn)`.

### Privacy baseline — `nndr`
The synthetic NNDR is the median over synthetic rows of `syn_to_real(1) / syn_self(1)` (as today).
With holdout, compute the same ratio for the holdout: median over holdout rows of
`holdout_to_real(1) / holdout_self(1)`. Report `nndr_median` and `nndr_median_holdout`;
**`score = clamp(nndr_median_synthetic / (nndr_median_holdout + eps))`** — a synthetic whose
nearest-real/nearest-self ratio is at least as large as the holdout's (i.e. no closer to real than a
genuine held-out sample) scores 1.0; smaller means risk. Without holdout: today's absolute score.

## UI

- An optional third `st.file_uploader` ("Holdout / test data (CSV/Parquet)") in `_sidebar()`; loaded via
  `core.load_table`. `core.run_report` gains a `holdout` param forwarded to `evaluate_with`.
- If a holdout is uploaded with a column set different from real → the existing `SynevaError` surfaces as
  an inline `st.error` (already handled by the run-time try/except).

## Testing

- **Behaviour preservation:** with `holdout=None`, every one of the five metrics' existing tests passes
  UNCHANGED and the golden snapshot stays green.
- **`encode_frames`:** three frames encode into a shared space (same column width; a row identical across
  frames maps to the same vector).
- **`Neighbors.holdout_to_real`:** euclidean equivalence on a sample; raises when no holdout given.
- **Utility holdout:** TSTR/TRTR train on full data and test on the provided holdout (assert it runs and
  the ratio is sensible; identical real==syn with a holdout → ratio near 1).
- **MIA holdout:** members=real vs non-members=holdout yields an AUC in [0,1]; a leaky synthetic (copies of
  real) makes members more identifiable than holdout (auc > 0.5 → lower score).
- **DCR/NNDR baseline:** a synthetic that is a verbatim copy of real, with a disjoint holdout, scores low
  (synthetic far closer than holdout); a synthetic drawn like the holdout scores near 1. Baseline scalars
  present.
- **Integration:** `evaluate(real, syn, meta, tiers=("core","extended"), holdout=hold, run_utility=True,
  utility_tasks=[...])` runs error-free; a column-mismatch holdout raises `SynevaError`.
- 85% coverage gate holds.

## Open questions
None outstanding. DCR and NNDR baseline scores are both relative ratios (synthetic statistic / holdout
statistic, clamped to [0,1]); both default to today's absolute behaviour when no holdout is supplied.
