# syneva — Component C4-c: UI benchmark tab — Design

**Date:** 2026-06-30
**Status:** Approved (pending spec review)
**Depends on:** C4-a (`syneva.benchmark`, `BenchmarkResult`), C4-b (`render_leaderboard`,
`BenchmarkResult.to_html`/`to_json`). Reuses the existing Streamlit UI (`src/syneva/ui/app.py`,
`src/syneva/ui/core.py`): `load_table`, `metadata_rows`/`metadata_from_editor`, `report_html_str`/
`report_json_str`, `run_report`, `_sidebar`/`_render_results`/`main`.
**Roadmap:** component C4 (benchmark & ranking engine), sub-spec C4-c (UI). Completes C4.

## Goal

Add a benchmark mode to the Streamlit UI: upload real data + multiple synthetic candidates, run
`syneva.benchmark`, and render the leaderboard inline with downloads. A sidebar mode radio switches
between the existing single-scorecard flow (left byte-identical) and the new benchmark flow.

## Scope

**In scope:** a sidebar "Mode" radio; a benchmark sidebar (real + multi-candidate upload + holdout +
metadata editor + preset + normalization + run button); a benchmark results renderer (inline
leaderboard + ranked summary + downloads); three thin `core` helpers; `main()` dispatch.

**Out of scope:** PDF leaderboard in the UI; saving/loading benchmark sessions; live normalization
re-render without a Streamlit rerun (Streamlit reruns on widget change anyway).

## `core.py` additions (thin wrappers; engine/render logic unchanged)

```python
def run_benchmark(
    real: pd.DataFrame | None,
    candidates: dict[str, pd.DataFrame],
    metadata: Metadata,
    *,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
) -> BenchmarkResult:
    """Run the benchmark engine for the UI (mirrors run_report's role)."""
    import syneva
    return syneva.benchmark(real, candidates, metadata, holdout=holdout, preset=preset)


def leaderboard_html_str(result: BenchmarkResult, *, normalization: str = "absolute") -> str:
    """In-memory leaderboard HTML (analog of report_html_str)."""
    from syneva.render.html.benchmark_renderer import render_leaderboard
    return render_leaderboard(result, normalization=normalization)


def benchmark_json_str(result: BenchmarkResult) -> str:
    import json
    return json.dumps(result.to_dict(), indent=2, default=str)
```

- When no `preset` is chosen in the UI, pass `preset=None` (engine resolves to defaults); the benchmark
  UI does not expose the manual per-metric selection (that's the single-scorecard mode's feature) — it
  uses presets for the shared config, defaulting to the engine default (core tier) when "Custom"/none.
- `run_benchmark` deliberately omits `tiers/cs/run_utility/...` for now; preset covers the common cases.
  (If "Custom" is selected we simply pass `preset=None`, i.e. the default core scorecard.)

## `app.py` additions

### Mode radio (top of sidebar, in `main()` or a tiny helper)
```python
mode = st.sidebar.radio("Mode", ["Single scorecard", "Benchmark (compare datasets)"])
```
`main()` dispatches: "Single scorecard" → the EXISTING flow verbatim (call `_sidebar()` etc., unchanged);
"Benchmark (compare datasets)" → `_benchmark_sidebar()` + benchmark rendering.

### `_benchmark_sidebar() -> dict | None`
A `with st.sidebar:` block:
1. **Real data** — `st.file_uploader("Real data (CSV/Parquet)", type=["csv","parquet"])`; load via
   `core.load_table`. Optional but recommended; if absent, real-requiring metrics are skipped by the
   engine.
2. **Synthetic candidates** — `st.file_uploader("Synthetic candidates", type=["csv","parquet"],
   accept_multiple_files=True)`. Build `candidates: dict[name, df]` where `name` = the uploaded file's
   stem (`Path(file.name).stem`); on a duplicate stem, append `_2`, `_3`, … to keep names unique.
3. **Holdout** — optional `st.file_uploader("Holdout / test data (optional)", ...)`.
4. **Metadata** — infer from real (or the first candidate if no real) via `core.metadata_rows`, show
   the existing `st.data_editor`, convert with `core.metadata_from_editor` at run time (reuse the
   single-mode helpers exactly).
5. **Profile** — `st.selectbox("Profile", ["Custom", "fast", "full", "privacy"])`; `None` if "Custom".
6. **Normalization** — `st.selectbox("Ranking normalization", ["absolute","linear","normal","quantile"])`.
7. **Run** — `st.button("Run benchmark", type="primary")`.

Returns `{"real", "candidates", "holdout", "edited", "preset", "normalization", "run"}` or `None` if
no candidate files yet.

### `_render_benchmark(result, normalization)`
- A ranked summary first: `st.dataframe` of `[{"rank","candidate","overall", **per-C}]` from
  `result.ranking(normalization)` + `result.overall`/`result.c_scores` (quick at-a-glance table).
- The full leaderboard inline: `components.html(core.leaderboard_html_str(result,
  normalization=normalization), height=900, scrolling=True)`.
- Downloads: `st.download_button("Download HTML", core.leaderboard_html_str(...), "leaderboard.html")`
  and `st.download_button("Download JSON", core.benchmark_json_str(result), "benchmark.json")`.

### `main()` dispatch
```python
def main():
    st.set_page_config(page_title="syneva", layout="wide")
    st.title("syneva — 7 Cs scorecard")
    mode = st.sidebar.radio("Mode", ["Single scorecard", "Benchmark (compare datasets)"])
    if mode == "Single scorecard":
        ... existing body verbatim ...
        return
    # Benchmark mode
    cfg = _benchmark_sidebar()
    if cfg and cfg["run"]:
        if not cfg["candidates"]:
            st.warning("Upload at least one synthetic candidate.")
        else:
            try:
                meta = core.metadata_from_editor(cfg["edited"])
                st.session_state["benchmark"] = (
                    core.run_benchmark(
                        cfg["real"], cfg["candidates"], meta,
                        holdout=cfg["holdout"], preset=cfg["preset"],
                    ),
                    cfg["normalization"],
                )
            except (SynevaError, ValueError) as e:
                st.error(f"Benchmark failed: {e}")
    bm = st.session_state.get("benchmark")
    if bm is None:
        st.write("Upload real data and synthetic candidates, then click **Run benchmark**.")
        return
    result, normalization = bm
    _render_benchmark(result, normalization)
```
(The existing single-mode `main()` body moves into the `if mode == "Single scorecard":` branch
unchanged. The `report` session key stays separate from the new `benchmark` key so switching modes
doesn't clobber either.)

## Error handling
- 0 candidates uploaded → `st.warning`, no run.
- Schema-mismatch candidate / unknown normalization → `SynevaError` from the engine/renderer, caught →
  `st.error`.
- A single candidate is allowed (degenerate ranking; the engine note surfaces in the leaderboard
  modeline).

## Testing
- **`core.run_benchmark`** returns a `BenchmarkResult` ranking `good` above `shifted` (fixtures).
- **`core.leaderboard_html_str`** returns HTML equal to `render_leaderboard(result, normalization=...)`
  and contains each candidate; **`core.benchmark_json_str`** round-trips via `BenchmarkResult.from_dict`.
- **Import-safe:** `import syneva.ui.app` succeeds (no Streamlit calls at import).
- **Single-mode preserved:** existing UI tests (`tests/unit/ui/...`) pass UNCHANGED.
- **Headless boot smoke:** launch on a test port → HTTP 200, no tracebacks, then kill.
- Full suite + golden green; 85% coverage.

## Open questions
None outstanding. Benchmark mode uses preset-based shared config (no per-metric manual selection);
candidate names derive from uploaded filename stems (deduped).
