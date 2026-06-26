# syneva — Component C4-b: leaderboard report + CLI — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)
**Depends on:** C4-a (`benchmark`, `BenchmarkResult` with `.reports`, `.score_matrix`, `.c_scores`,
`.overall`, `.metrics()`, `.c_dims()`, `.ranking(normalization, by)`, `.to_dict`/`from_dict`/`to_json`).
Reuses: `render_html`, the Jinja2 env, `metric_info` helpers (`display_name`, `describe_c`,
`humanize_scalar`, `verdict`), the CLI `_load_df`/`_load_metadata`/`_UNSET` plumbing.
**Roadmap:** component C4 (benchmark & ranking engine), sub-spec C4-b (report + CLI). Sibling remaining:
C4-c UI benchmark tab.

## Goal

Render a `BenchmarkResult` into a shareable HTML/JSON leaderboard and add a `syneva benchmark` CLI
command that takes a real dataset + N named synthetic candidates and writes the leaderboard. Pure
rendering + CLI on top of the C4-a engine — no engine logic changes (only thin `to_html` wrappers).

## Scope

**In scope:** a benchmark HTML renderer + Jinja2 template; `BenchmarkResult.to_html`; the
`syneva benchmark` CLI command (repeatable `--candidate name=path`, shared-config flags, normalization,
output files).

**Out of scope:** UI benchmark tab (C4-c); PDF leaderboard; manifest-file input (possible later).

## Rendering

New `src/syneva/render/html/benchmark_renderer.py` + `templates/leaderboard.html.j2`. Reuse the existing
`_ENV` Jinja2 environment (extend `renderer.py`'s env or construct an equivalent
`FileSystemLoader(_TEMPLATES)` — share the same `templates/` dir), the inline-CSS style conventions, the
`verdict()` color classes (`v-excellent`/`v-good`/`v-fair`/`v-poor`), and `display_name`/`describe_c`/
`humanize_scalar`.

```python
def render_leaderboard(
    result: BenchmarkResult, *, normalization: str = "absolute", interactive: bool = False
) -> str:
```

Builds a view and renders these sections:

1. **Leaderboard table** — from `result.ranking(normalization)`: columns `rank`, `candidate`,
   `overall` (verdict-colored, from `result.overall`), then one column per C in `result.c_dims()`
   (each cell the candidate's `c_scores[candidate][c]`, verdict-colored; blank if that C absent for the
   candidate). A modeline states the active `normalization` and clarifies: *displayed scores are
   absolute [0,1]; the ranking order reflects the selected normalization.*
2. **Score matrix** — candidates × metrics grid from `result.score_matrix`, rows = candidates (in
   leaderboard rank order), columns = `result.metrics()` (header via `display_name`), each cell
   verdict-colored; missing metric for a candidate → blank cell. This is the cross-candidate
   "who wins each metric" view.
3. **Per-C bars** — for each candidate, a compact horizontal CSS bar per C dimension (width =
   `score * 100%`, verdict-colored), giving the at-a-glance shape. Pure CSS (no matplotlib dependency).
4. **Per-candidate drill-down** — for each candidate, embed its existing scorecard rendered via
   `render_html(result.reports[candidate], interactive=interactive)` inside a collapsible
   `<details><summary>candidate</summary>…</details>` block. To embed safely, render each candidate's
   scorecard to its HTML string and inject the `<body>` inner content (or the full document inside an
   isolating wrapper). **Implementation note:** `render_html` returns a full `<!doctype html>` document;
   embedding full documents nested is invalid HTML. The renderer must extract the body/main content (a
   small helper that slices between `<body>` … `</body>`, or—cleaner—`render_html` gains an internal
   `fragment: bool = False` flag that, when True, renders only the inner card markup without the
   `<html>/<head>` chrome). **Choose the fragment-flag approach**: add `fragment` to `render_html`
   (default False ⇒ existing behavior unchanged, every current caller/test identical) and a matching
   `{% if not fragment %}`-guarded doctype/head/style block in `scorecard.html.j2`, so the leaderboard
   embeds fragments and pulls the shared `<style>` once into the leaderboard `<head>`.

### `BenchmarkResult.to_html` (engine.py — thin wrapper)

```python
def to_html(self, path, *, normalization: str = "absolute", interactive: bool = False) -> None:
    from syneva.render.html.benchmark_renderer import render_leaderboard
    Path(path).write_text(render_leaderboard(self, normalization=normalization, interactive=interactive))
```
Mirrors `Report.to_html`. `to_json`/`from_json` already exist from C4-a (unchanged).

## CLI — `syneva benchmark`

Add a `benchmark` command to `src/syneva/cli/main.py`:

```python
@app.command()
def benchmark(
    real: Path | None = typer.Option(None, exists=True, dir_okay=False),
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="named synthetic dataset as name=path (repeatable)"
    ),
    metadata: Path | None = typer.Option(None, exists=True, dir_okay=False),
    out: Path = typer.Option(Path("./syneva-benchmark")),
    holdout: Path | None = typer.Option(None, exists=True, dir_okay=False),
    preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
    tiers: str | None = typer.Option(None, help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    normalization: str = typer.Option("absolute", help="ranking: absolute|linear|normal|quantile"),
) -> None:
```

Behavior:
- Parse each `--candidate` entry as `name=path`: split on the first `=`; error via
  `typer.BadParameter` if there is no `=`, the name is empty, or the name repeats. Load each path with
  the existing `_load_df`; a missing/!exists path → `typer.BadParameter`.
- Build `candidates: dict[name, df]` preserving CLI order.
- `real_df = _load_df(real) if real else None`; `meta = _load_metadata(metadata)`;
  `holdout_df = _load_df(holdout) if holdout else None`.
- Map omitted shared-config options to `_UNSET` exactly as the `evaluate` command does
  (`tiers`/`cs` → tuple-or-`_UNSET`, `run_utility` → `True`-or-`_UNSET`); forward `preset=preset`.
- `res = syneva.benchmark(real_df, candidates, meta, holdout=holdout_df, preset=preset, tiers=…,
  cs=…, run_utility=…, ...)`.
- `out.mkdir(parents=True, exist_ok=True)`; `res.to_html(out / "leaderboard.html",
  normalization=normalization)`; `res.to_json(out / "benchmark.json")`.
- Echo the ranked order to stdout, e.g. `for rank, name, score in res.ranking(normalization):
  typer.echo(f"{rank}. {name}  {score:.4f}")`, then `typer.echo(f"wrote leaderboard to {out}/")`.
- Wrap the body in the existing `try/except SynevaError` → `typer.echo(..., err=True)` + exit code 2,
  matching the `evaluate` command. An unknown `normalization` surfaces as `SynevaError` from
  `ranking()` (exit 2).

## Error handling

- Malformed `--candidate` (no `=`, empty name, duplicate name) → `typer.BadParameter` (exit 2).
- Empty candidate set is impossible (the option is required, `...`); a single candidate is allowed
  (benchmark/ranking already handle the degenerate cohort with a note).
- Schema-mismatch candidate → `SynevaError` from `benchmark()` (exit 2).
- Unknown `normalization` → `SynevaError` from `ranking()`/`render_leaderboard` (exit 2).

## Testing

**Renderer:**
- `render_leaderboard(res)` returns a non-empty HTML string containing each candidate name, the
  leaderboard table, a per-C column header for each `c_dims()`, and score-matrix cells for the metrics.
- Rank order in the rendered table matches `res.ranking(normalization)` for each normalization mode;
  the modeline names the active normalization.
- An all-errored candidate appears LAST in the table (consistent with C4-a ranking).
- `interactive=True` vs `False` both render without error.
- **Fragment flag behavior-preservation:** `render_html(report)` (default `fragment=False`) output is
  UNCHANGED — existing scorecard tests and the golden HTML (if any) stay green;
  `render_html(report, fragment=True)` omits the `<!doctype html>`/`<head>` chrome.
- `BenchmarkResult.to_html(tmp_path/"lb.html")` writes a file whose content equals
  `render_leaderboard(res)`.

**CLI:**
- `syneva benchmark --real r --candidate good=g --candidate shifted=s --out O` exits 0 and writes
  `O/leaderboard.html` and `O/benchmark.json`; stdout lists the ranked candidates with `good` first
  (using the good/shifted fixtures).
- Malformed `--candidate good` (no `=`) → non-zero exit; duplicate name → non-zero exit.
- `--normalization linear` honored (order + modeline); `--normalization bogus` → non-zero exit.
- `--preset privacy` yields a leaderboard whose per-C columns are compliance-only.
- Reuses fixtures; subprocess or `CliRunner` matching the existing CLI test style.

**Full suite / golden:** additive; `render_html` default path unchanged ⇒ golden stays green
(NEVER `--update-golden`). 85% coverage gate holds.

## Open questions
None outstanding. Per-candidate scorecards are embedded as collapsible `<details>` fragments (via a new
`render_html(fragment=True)` flag) rather than separate linked files, for a single self-contained
artifact.
