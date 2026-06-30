# UI Benchmark Tab (C4-c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a benchmark mode to the Streamlit UI — upload real data + multiple synthetic candidates, run `syneva.benchmark`, render the leaderboard inline with downloads — selected by a sidebar mode radio that leaves the existing single-scorecard flow untouched.

**Architecture:** Three thin `core` helpers (`run_benchmark`, `leaderboard_html_str`, `benchmark_json_str`) mirror the existing `run_report`/`report_html_str`/`report_json_str`. `app.py` gains a sidebar mode radio, a `_benchmark_sidebar()`, a `_render_benchmark()`, and a `main()` dispatch; the current single-mode body is moved verbatim into the "Single scorecard" branch.

**Tech Stack:** Python 3.10+, streamlit, pandas, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-30-syneva-benchmark-4c-ui.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Behavior preservation:** the single-scorecard flow must behave exactly as today. `import syneva.ui.app` must stay import-safe (no Streamlit calls at import time — all UI calls live inside functions). Existing UI tests and the golden snapshot stay green (NEVER `--update-golden`). If an existing test or golden changes, STOP and report.
- Reuse, don't reimplement: `core.load_table`, `core.metadata_rows`, `core.metadata_from_editor`, `core.run_report`; the engine `syneva.benchmark` and `render_leaderboard` (C4-a/b).

## File structure
- Modify: `src/syneva/ui/core.py` (add 3 helpers)
- Modify: `src/syneva/ui/app.py` (mode radio, `_benchmark_sidebar`, `_render_benchmark`, `main` dispatch)
- Tests: `tests/unit/ui/test_core.py` (append), `tests/unit/ui/test_app_importable.py` (confirm still passes)

---

## Task 1: `core` benchmark helpers

**Files:** Modify `src/syneva/ui/core.py`; Test append `tests/unit/ui/test_core.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/ui/test_core.py
def test_run_benchmark_ranks_good_first():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    shifted = pd.read_parquet(f"{FIX}/adult_income_syn_shifted_500.parquet")
    res = core.run_benchmark(
        real, {"good": good, "shifted": shifted}, Metadata.infer(real)
    )
    ranked = [name for _, name, _ in res.ranking()]
    assert ranked[0] == "good"


def test_leaderboard_html_str_matches_renderer():
    from syneva.render.html.benchmark_renderer import render_leaderboard

    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    res = core.run_benchmark(real, {"good": good}, Metadata.infer(real))
    html = core.leaderboard_html_str(res, normalization="linear")
    assert html == render_leaderboard(res, normalization="linear")
    assert "good" in html


def test_benchmark_json_str_roundtrips():
    from syneva.benchmark.engine import BenchmarkResult

    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    good = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    res = core.run_benchmark(real, {"good": good}, Metadata.infer(real))
    s = core.benchmark_json_str(res)
    import json
    restored = BenchmarkResult.from_dict(json.loads(s))
    assert restored.ranking() == res.ranking()
```

(Confirm `FIX`, `core`, `Metadata`, `pd` are already imported in `tests/unit/ui/test_core.py`; reuse them. Confirm the shifted fixture filename.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/ui/test_core.py -k "benchmark or leaderboard" --no-cov -v` (AttributeError: no `run_benchmark`).

- [ ] **Step 3: Implement** — in `src/syneva/ui/core.py`, add three functions near the existing `report_html_str`/`report_json_str` (the file already imports `json` and `pd`; confirm and reuse):

```python
def run_benchmark(
    real,
    candidates: dict,
    metadata,
    *,
    holdout=None,
    preset: str | None = None,
):
    """Run the benchmark engine for the UI (mirrors run_report's role)."""
    import syneva

    return syneva.benchmark(real, candidates, metadata, holdout=holdout, preset=preset)


def leaderboard_html_str(result, *, normalization: str = "absolute") -> str:
    """In-memory leaderboard HTML (analog of report_html_str)."""
    from syneva.render.html.benchmark_renderer import render_leaderboard

    return render_leaderboard(result, normalization=normalization)


def benchmark_json_str(result) -> str:
    """Serialize a BenchmarkResult to a JSON string."""
    return json.dumps(result.to_dict(), indent=2, default=str)
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/ui/test_core.py --no-cov -v` (existing + 3 new pass).

- [ ] **Step 5: Commit** — `git commit -m "feat(ui): core benchmark helpers (run_benchmark/leaderboard_html_str/benchmark_json_str)"`

---

## Task 2: `app.py` mode radio, benchmark sidebar, render, dispatch

**Files:** Modify `src/syneva/ui/app.py`; Test: `tests/unit/ui/test_app_importable.py` (must still pass) + boot smoke.

- [ ] **Step 1: Confirm the import-safety test exists** — read `tests/unit/ui/test_app_importable.py`. It should assert `import syneva.ui.app` works (no Streamlit execution at import). If it asserts a specific symbol, keep that intact. No new test file is required for this task beyond the boot smoke (Streamlit widget flows aren't unit-testable without a heavy harness); the import test + headless boot are the gates. (If `tests/unit/ui/test_app_importable.py` does not exist, create it with: `def test_app_module_imports(): import syneva.ui.app  # noqa: F401`.)

- [ ] **Step 2: Implement** — edit `src/syneva/ui/app.py`. Do NOT change `_sidebar()` or `_render_results()`. Add the imports needed (the file already imports `streamlit as st`, `streamlit.components.v1 as components`, `core`, `SynevaError`, `Metadata`; add `from pathlib import Path` at the top with the other imports).

(a) Add `_benchmark_sidebar()` (place after `_render_results`, before `main`):
```python
def _benchmark_sidebar() -> dict | None:
    """Sidebar controls for benchmark mode; returns a config dict or None."""
    with st.sidebar:
        st.header("1 · Real data")
        real_file = st.file_uploader(
            "Real data (CSV/Parquet)", type=["csv", "parquet"], key="bm_real"
        )
        st.header("2 · Synthetic candidates")
        cand_files = st.file_uploader(
            "Synthetic candidates (CSV/Parquet)",
            type=["csv", "parquet"],
            accept_multiple_files=True,
            key="bm_candidates",
        )
        holdout_file = st.file_uploader(
            "Holdout / test data (optional)", type=["csv", "parquet"], key="bm_holdout"
        )

        real = core.load_table(real_file) if real_file is not None else None
        candidates: dict = {}
        for f in cand_files or []:
            name = Path(f.name).stem
            unique = name
            i = 2
            while unique in candidates:
                unique = f"{name}_{i}"
                i += 1
            candidates[unique] = core.load_table(f)
        holdout = core.load_table(holdout_file) if holdout_file is not None else None

        if not candidates:
            st.info("Upload one or more synthetic candidate files to begin.")
            return None

        st.header("3 · Column metadata")
        basis = real if real is not None else next(iter(candidates.values()))
        edited = st.data_editor(
            core.metadata_rows(Metadata.infer(basis)),
            num_rows="fixed",
            use_container_width=True,
            key="bm_meta",
        )

        st.header("4 · Profile & ranking")
        profile = st.selectbox(
            "Profile", ["Custom", "fast", "full", "privacy"], index=0, key="bm_profile"
        )
        normalization = st.selectbox(
            "Ranking normalization",
            ["absolute", "linear", "normal", "quantile"],
            index=0,
            key="bm_norm",
        )
        run = st.button("Run benchmark", type="primary", key="bm_run")

    return {
        "real": real,
        "candidates": candidates,
        "holdout": holdout,
        "edited": edited,
        "preset": None if profile == "Custom" else profile,
        "normalization": normalization,
        "run": run,
    }
```

(b) Add `_render_benchmark(result, normalization)` (after `_benchmark_sidebar`):
```python
def _render_benchmark(result, normalization: str) -> None:
    st.subheader("Leaderboard")
    overall = result.overall
    c_scores = result.c_scores
    summary = []
    for rank, name, _key in result.ranking(normalization=normalization):
        row = {"rank": rank, "candidate": name, "overall": round(overall.get(name, 0.0), 3)}
        for c, score in c_scores.get(name, {}).items():
            row[c] = round(score, 3)
        summary.append(row)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    html_str = core.leaderboard_html_str(result, normalization=normalization)
    components.html(html_str, height=900, scrolling=True)
    st.download_button("Download HTML", html_str, "leaderboard.html")
    st.download_button("Download JSON", core.benchmark_json_str(result), "benchmark.json")
```

(c) Rewrite `main()` to dispatch on a mode radio. The CURRENT `main()` body (set_page_config, title, `cfg = _sidebar()`, the run/warn/error block, the `report = st.session_state.get("report")` rendering) becomes the "Single scorecard" branch VERBATIM:
```python
def main() -> None:
    st.set_page_config(page_title="syneva", layout="wide")
    st.title("syneva — 7 Cs scorecard")
    mode = st.sidebar.radio(
        "Mode", ["Single scorecard", "Benchmark (compare datasets)"], key="mode"
    )

    if mode == "Single scorecard":
        cfg = _sidebar()
        if cfg and cfg["run"]:
            if cfg["preset"] is None and not cfg["selected"]:
                st.warning("Select at least one evaluator, or choose a Profile.")
            elif cfg["utility_selected"] and not cfg["utility_tasks"]:
                st.warning("A utility metric is selected but no target column was chosen.")
            elif cfg["fairness_selected"] and not cfg["fairness_specs"]:
                st.warning(
                    "A fairness metric is selected but no protected attribute/outcome was chosen."
                )
            else:
                try:
                    meta = core.metadata_from_editor(cfg["edited"])
                    st.session_state["report"] = core.run_report(
                        cfg["real"],
                        cfg["synthetic"],
                        meta,
                        cfg["selected"],
                        cfg["utility_tasks"],
                        fairness_specs=cfg["fairness_specs"],
                        holdout=cfg["holdout"],
                        preset=cfg["preset"],
                    )
                except (SynevaError, ValueError) as e:
                    st.error(f"Evaluation failed: {e}")
        report = st.session_state.get("report")
        if report is None:
            st.write("Configure inputs in the sidebar, then click **Run evaluation**.")
            return
        _render_results(report)
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
                        cfg["real"],
                        cfg["candidates"],
                        meta,
                        holdout=cfg["holdout"],
                        preset=cfg["preset"],
                    ),
                    cfg["normalization"],
                )
            except (SynevaError, ValueError) as e:
                st.error(f"Benchmark failed: {e}")
    bm = st.session_state.get("benchmark")
    if bm is None:
        st.write(
            "Upload real data and synthetic candidates in the sidebar, then click **Run benchmark**."
        )
        return
    result, normalization = bm
    _render_benchmark(result, normalization)
```

IMPORTANT: copy the existing `main()` single-mode body EXACTLY (same warnings, same `run_report` args, same messages) into the `if mode == "Single scorecard":` branch — do not alter its behavior. Verify against the current file before editing.

- [ ] **Step 3: Import-safety + targeted** — `uv run pytest tests/unit/ui/ --no-cov -v` (all pass, including `test_app_importable`).

- [ ] **Step 4: Headless boot smoke** — `uv run streamlit run src/syneva/ui/app.py --server.headless true --server.port 8606 &`, wait ~6s, `curl -s -o /dev/null -w "%{http_code}" http://localhost:8606` (expect 200), check the streamlit log for tracebacks, then kill the backgrounded PID. If streamlit isn't installed, note it and skip ONLY the boot smoke.

- [ ] **Step 5: FULL SUITE (behavior-preservation gate)** — `uv run pytest -q`. ALL pass (2 pre-existing skips); golden UNCHANGED; coverage >= 85%. If golden fails, STOP and report BLOCKED.

- [ ] **Step 6: Commit** — `uv run ruff check --fix . && uv run ruff format .`, `git add -A`, `git commit -m "feat(ui): benchmark mode (mode radio, multi-candidate upload, inline leaderboard)"`. Re-run with `git add -A` if a hook aborts.

---

## Self-review notes
- **Spec coverage:** core helpers run_benchmark/leaderboard_html_str/benchmark_json_str (T1) · mode radio + _benchmark_sidebar (multi-file candidates w/ stem-name dedup, holdout, metadata reuse, preset, normalization) + _render_benchmark (ranked dataframe + inline leaderboard + downloads) + main dispatch with single-mode body verbatim (T2). Import-safety + boot smoke + golden gate in T2.
- **Type/signature consistency:** `core.run_benchmark(real, candidates, metadata, *, holdout, preset)` matches `syneva.benchmark`; `leaderboard_html_str(result, *, normalization)` matches `render_leaderboard`; `_render_benchmark` reads `ranking`/`overall`/`c_scores` (all exist on BenchmarkResult); session keys `report` vs `benchmark` kept separate so modes don't clobber.
- **No placeholders:** complete code for every helper, the sidebar, the renderer, and main dispatch; single-mode body reproduced verbatim.
- **Behavior preservation:** `_sidebar`/`_render_results` untouched; single-mode `main` body copied exactly; only additive code paths; golden + import-safety + existing UI tests gate it.
