# syneva — Phase 0c-a: Gower distance backend — Design

**Date:** 2026-06-23
**Status:** Approved (pending spec review)
**Depends on:** syneva v0.1 + 0a + 0b. Public API: `registry`, `MetricSpec`, `MetricResult`, `Metadata`,
`ColumnType`, `encode_pair`, `evaluate`/`evaluate_with`, the `_instantiate` config pattern.
**Roadmap:** component C3 (methodology rigor), split per the 2026-06-23 decision into focused specs;
this is the first — the Gower distance backend (SynthEval's signature method). Siblings (holdout,
presets, multi-classifier utility) are separate specs.

## Goal

Add an optional **Gower distance** backend so the seven nearest-neighbour metrics can treat mixed
numeric/categorical data the way SynthEval does, without one-hot encoding or standardization. Euclidean
(the current behaviour) stays the default and is **exactly preserved**; Gower is opt-in via
`evaluate(..., distance="gower")`. A shared neighbour helper removes the duplicated sklearn calls now
copy-pasted across the metrics.

## Scope

**In scope:**
- `core/distance.py`: `gower_matrix(A, B, meta)`.
- `core/neighbors.py`: a `Neighbors` helper exposing the neighbour primitives the metrics need, with
  euclidean and gower backends.
- Refactor the 7 NN metrics to use `Neighbors` and accept a `distance` field.
- `distance="euclidean"|"gower"` on `evaluate`/`evaluate_with`, passed via `_instantiate`.

**Out of scope (separate specs):** holdout test-set, presets, multi-classifier utility. Entropy-weighted
identifiability remains future work. Gower is not applied to non-NN metrics (they don't use distances).

## The seven NN metrics and the primitives they need

| metric | neighbour calls (in `Neighbors` terms) |
|---|---|
| `dcr` | `syn_to_real(1)` → median + 5th-percentile |
| `nndr` | `syn_to_real(1)` / `syn_self(1)` |
| `nn_adversarial_accuracy` | `real_self(1)`, `syn_self(1)`, `real_to_syn(1)`, `syn_to_real(1)` |
| `authenticity` | `syn_to_real(1)`, `real_self(1)` |
| `alpha_precision_beta_recall` | `real_self(k)`, `syn_self(k)`, `syn_to_real(1)`, `real_to_syn(1)` |
| `epsilon_identifiability` | `real_self(1)`, `real_to_syn(1)` |
| `attribute_disclosure` | `real_to_syn_index()` (built on the QI-restricted metadata) |

## `core/distance.py` — `gower_matrix(A, B, meta) -> np.ndarray`

Returns an `(len(A), len(B))` matrix of Gower distances in `[0, 1]`. Per feature in `meta.columns`
(numeric or categorical; others ignored):
- **Numeric:** `|a − b| / range`, where `range = (combined max − combined min)` over `A∪B` for that
  column (`range==0` → contributes 0). NaNs contribute 0 (treated as no disagreement) — documented.
- **Categorical:** `0` if equal else `1` (NaN normalized to a `"__NA__"` token first, matching `encode_pair`).
- The distance is the **mean over contributing features** (equal weights). If no usable features, returns
  an all-zeros matrix.

Implementation is vectorized per feature (broadcast `A_col[:, None]` vs `B_col[None, :]`), accumulating a
sum and a feature count, then dividing.

## `core/neighbors.py` — the `Neighbors` helper

```python
class Neighbors:
    def __init__(self, real, synthetic, meta, *, distance="euclidean", cap=2000, random_state=42): ...
    def real_self(self, k=1) -> np.ndarray         # dist to k-th nearest OTHER real, per real row
    def syn_self(self, k=1) -> np.ndarray
    def real_to_syn(self, k=1) -> np.ndarray        # per real row, dist to k-th nearest synthetic
    def syn_to_real(self, k=1) -> np.ndarray
    def real_to_syn_index(self) -> np.ndarray       # per real row, index of nearest synthetic (k=1)
    @property
    def n_features(self) -> int                     # encodable/usable feature count (0 => degenerate)
```

- **Euclidean backend:** call `encode_pair(real, synthetic, meta)` once to get `X_real, X_syn`; build
  sklearn `NearestNeighbors` per query as the current metrics do. `real_self(k)` =
  `NearestNeighbors(k+1).fit(X_real).kneighbors(X_real)[0][:, k]`; `real_to_syn(k)` =
  `NearestNeighbors(k).fit(X_syn).kneighbors(X_real)[0][:, k-1]`; `real_to_syn_index()` =
  `NearestNeighbors(1).fit(X_syn).kneighbors(X_real, return_distance=False)[:, 0]`; analogously for the
  syn-anchored queries. `n_features = X_real.shape[1]`. **No capping** (preserves current behaviour exactly).
- **Gower backend:** if either frame exceeds `cap`, take a seeded (`default_rng(random_state)`) subsample
  and record that it was capped (exposed via a `notes`-style attribute the metrics can surface).
  Precompute `D_rr = gower_matrix(real, real, meta)`, `D_rs = gower_matrix(real, synthetic, meta)`,
  `D_ss = gower_matrix(synthetic, synthetic, meta)`. `real_self(k)`: set the diagonal of `D_rr` to `inf`,
  take the k-th smallest per row (`np.partition`). `real_to_syn(k)`: k-th smallest of `D_rs` per row.
  `syn_to_real(k)`: k-th smallest of `D_rs.T` per row. `real_to_syn_index()`: `argmin(D_rs, axis=1)`.
  `n_features` = count of usable columns in `meta`.

Both backends must handle the degenerate cases the metrics already guard (`n_features == 0`, too-few
rows) — the metrics keep their existing guards, now expressed via `neighbors.n_features` / array lengths.

## Metric refactor (behaviour-preserving for euclidean)

Each of the 7 metrics:
1. Gains a `distance: str = "euclidean"` field (the privacy/coverage metrics become small dataclasses or
   keep their plain-class form with a `distance` attribute set in `__init__`; match each file's existing
   shape — most are plain classes, so add an `__init__(self, distance="euclidean")`).
2. Builds `nb = Neighbors(real, synthetic, meta, distance=self.distance)` (attribute_disclosure builds it
   with `qi_meta`) and replaces its direct `encode_pair` + `NearestNeighbors` calls with the `nb.*`
   primitives in the table above. **The downstream score math is unchanged.**
3. Surfaces a capped note when `nb` reports capping (gower only).

**Hard constraint:** with `distance="euclidean"` (the default), every metric must produce byte-identical
results to today. The existing per-metric unit tests and the golden snapshot are the guardrail — they
must pass **unchanged**. (The euclidean `Neighbors` path issues the same sklearn calls on the same
`encode_pair` matrices, so results are identical.)

## Runner threading

- `evaluate()` / `evaluate_with()` gain `distance: str = "euclidean"` (validated against
  `{"euclidean", "gower"}`; bad value → `SynevaError`).
- Generalize `_instantiate` to also pass `distance=` to any metric whose constructor accepts it
  (alongside `tasks`/`specs`/`random_state`).
- The 7 NN metrics accept `distance`; all other metrics ignore it (signature-based passing already
  handles this).

## Testing

- **`gower_matrix`:** identical rows → 0; a hand-computed mixed-type example (one numeric + one
  categorical) matches the expected mean; constant numeric column contributes 0; NaN handling.
- **`Neighbors` euclidean-equivalence:** on a sample, `real_self(1)`, `real_to_syn(1)`, `syn_to_real(1)`,
  `real_to_syn_index()` equal the direct sklearn computation (guarantees behaviour preservation).
- **`Neighbors` gower:** sensible ordering on a tiny hand-built example.
- **Per-metric gower smoke tests:** each of the 7 metrics with `distance="gower"` on mixed-type data
  returns a valid score in `[0, 1]` with the expected direction (e.g. `dcr` higher for far synthetic).
- **Integration:** `evaluate(real, syn, meta, tiers=("core","extended"), distance="gower")` runs all NN
  metrics error-free; an invalid `distance` raises `SynevaError`.
- **Regression:** the full existing suite (euclidean default) + golden snapshot pass **unchanged**; 85%
  coverage gate holds.

## Open questions
None outstanding. Gower row cap = 2000 (matches `hitting_rate`); per-feature range over `A∪B`;
equal feature weights (SynthEval-style, unweighted).
