# Leaderboard Report + CLI (C4-b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a `BenchmarkResult` into a shareable HTML/JSON leaderboard and add a `syneva benchmark` CLI command.

**Architecture:** A `render_html(fragment=...)` flag lets each candidate's existing scorecard embed inside a new `leaderboard.html.j2` (ranked table + score matrix + per-C bars + collapsible drill-down), rendered by `benchmark_renderer.render_leaderboard`; `BenchmarkResult.to_html` wraps it; the CLI `benchmark` command parses `--candidate name=path` and writes the artifacts.

**Tech Stack:** Python 3.10+, jinja2, typer, pandas, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-06-26-syneva-benchmark-4b-report-cli.md`.

---

## CRITICAL environment notes
- Use `uv`. Targeted tests MUST use `--no-cov`. Before committing: `uv run ruff check --fix . && uv run ruff format .`, then `git add -A`, commit; if a hook modifies files and aborts, `git add -A` and re-run.
- **Behavior preservation:** `render_html(report)` with the default `fragment=False` MUST produce byte-identical output to today. Existing scorecard tests + the golden snapshot stay green (NEVER `--update-golden`). If an existing test or golden changes, STOP and report.
- Reuse, don't reimplement: `render_html` + `_ENV` live in `src/syneva/render/html/renderer.py`; metric-info helpers (`display_name`, `describe_c`, `humanize_scalar`, `verdict`) in `src/syneva/core/metric_info.py`; CLI helpers (`_load_df`, `_load_metadata`, `_UNSET`) in `src/syneva/cli/main.py`.

## File structure
- Modify: `src/syneva/render/html/renderer.py` (add `fragment` param), `src/syneva/render/html/templates/scorecard.html.j2` (guard chrome)
- Create: `src/syneva/render/html/benchmark_renderer.py`, `src/syneva/render/html/templates/leaderboard.html.j2`
- Modify: `src/syneva/benchmark/engine.py` (`BenchmarkResult.to_html`), `src/syneva/cli/main.py` (`benchmark` command)
- Tests: `tests/unit/render/test_fragment.py`, `tests/unit/render/test_leaderboard.py`, `tests/cli/test_benchmark_cmd.py`

---

## Task 1: `render_html(fragment=...)` flag (behavior-preserving)

**Files:** Modify `src/syneva/render/html/renderer.py`, `src/syneva/render/html/templates/scorecard.html.j2`; Test `tests/unit/render/test_fragment.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/render/test_fragment.py
import pandas as pd

import syneva
from syneva.render.html.renderer import render_html


def _report():
    real = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    syn = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    return syneva.evaluate(real, syn, syneva.Metadata.infer(real))


def test_default_is_full_document():
    html = render_html(_report())
    assert html.lstrip().startswith("<!doctype html")
    assert "</html>" in html


def test_fragment_omits_doctype_and_head():
    html = render_html(_report(), fragment=True)
    assert "<!doctype html" not in html.lower()
    assert "<head>" not in html.lower()
    assert "</html>" not in html.lower()
    # still contains the actual scorecard content
    assert "syneva scorecard" in html or "summary" in html
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/render/test_fragment.py --no-cov -v` (TypeError: unexpected `fragment`). You may need to create `tests/unit/render/` (no `__init__.py` — importlib mode).

- [ ] **Step 3: Implement**

(a) `src/syneva/render/html/renderer.py` — add `fragment: bool = False` to `render_html`'s signature (after `interactive`) and pass it to the template render call:
```python
def render_html(
    report: Report, *, interactive: bool = False, score_mode: str = "normalized", fragment: bool = False
) -> str:
```
and in the final `tpl.render(...)`, add `fragment=fragment,` to the kwargs.

(b) `src/syneva/render/html/templates/scorecard.html.j2` — wrap the document chrome so a fragment renders only the inner content. The current file is: lines 1-43 doctype/`<html>`/`<head>`/`<style>…</style>`/`</head>`, line 44 `<body>`, body content, line 129 `</body>`, line 130 `</html>`. Change so:
  - Wrap lines 1-44 (everything from `<!doctype html>` through `<body>`) in `{% if not fragment %}…{% endif %}`.
  - Wrap lines 129-130 (`</body>` and `</html>`) in `{% if not fragment %}…{% endif %}`.

Concretely, the top becomes:
```jinja
{% if not fragment %}
<!doctype html>
<html lang="en">
<head>
... (unchanged head + full <style>…</style>) ...
</head>
<body>
{% endif %}
  <h1>syneva scorecard</h1>
... (unchanged body content) ...
{% if not fragment %}
</body>
</html>
{% endif %}
```
Do NOT change any markup inside the body or the `<style>` contents — only add the two `{% if not fragment %}` guards. With `fragment` undefined/False (every existing caller), the template renders exactly as before (jinja treats an undefined `fragment` as falsy, but we pass it explicitly now).

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/unit/render/test_fragment.py --no-cov -v` (2 passed).

- [ ] **Step 5: Behavior-preservation gate** — `uv run pytest -q`. ALL pass (2 pre-existing skips); golden UNCHANGED; existing render/scorecard tests pass UNCHANGED. If the golden HTML/JSON differs, STOP — the default path changed.

- [ ] **Step 6: Commit** — `git commit -m "feat(render): render_html fragment flag (embeddable scorecard body)"`

---

## Task 2: `render_leaderboard` + template + `BenchmarkResult.to_html`

**Files:** Create `src/syneva/render/html/benchmark_renderer.py`, `src/syneva/render/html/templates/leaderboard.html.j2`; Modify `src/syneva/benchmark/engine.py`; Test `tests/unit/render/test_leaderboard.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/render/test_leaderboard.py
import pandas as pd

import syneva
from syneva.render.html.benchmark_renderer import render_leaderboard


def _result():
    real = pd.read_parquet("tests/fixtures/adult_income_real_500.parquet")
    good = pd.read_parquet("tests/fixtures/adult_income_syn_good_500.parquet")
    shifted = pd.read_parquet("tests/fixtures/adult_income_syn_shifted_500.parquet")
    meta = syneva.Metadata.infer(real)
    return syneva.benchmark(real, {"good": good, "shifted": shifted}, meta)


def test_leaderboard_contains_candidates_and_sections():
    html = render_leaderboard(_result())
    assert html.lstrip().startswith("<!doctype html")
    assert "good" in html and "shifted" in html
    # leaderboard table + score matrix + per-C bars present (by class hooks)
    assert "leaderboard" in html.lower()
    assert "score-matrix" in html.lower()


def test_leaderboard_rank_order_matches_ranking():
    res = _result()
    html = render_leaderboard(res, normalization="linear")
    ranked = [name for _, name, _ in res.ranking(normalization="linear")]
    # the first-ranked candidate's name appears before the second in the document
    assert html.index(ranked[0]) < html.index(ranked[1])
    assert "linear" in html.lower()  # modeline names the normalization


def test_leaderboard_embeds_candidate_scorecards():
    html = render_leaderboard(_result())
    # collapsible drill-down per candidate
    assert "<details" in html.lower()
    # embedded fragment content (a metric/section from the scorecard) is present
    assert "summary" in html.lower()


def test_to_html_writes_file(tmp_path):
    res = _result()
    p = tmp_path / "lb.html"
    res.to_html(p)
    assert p.read_text() == render_leaderboard(res)
```

(Confirm the shifted fixture filename `adult_income_syn_shifted_500.parquet` exists under `tests/fixtures/`; if the name differs use the real one.)

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/unit/render/test_leaderboard.py --no-cov -v` (ImportError).

- [ ] **Step 3: Implement the renderer** — create `src/syneva/render/html/benchmark_renderer.py`:

```python
from __future__ import annotations

from syneva.core.metric_info import describe_c, display_name, verdict
from syneva.render.html.renderer import _ENV, render_html


def render_leaderboard(
    result, *, normalization: str = "absolute", interactive: bool = False
) -> str:
    ranked = result.ranking(normalization=normalization)  # [(rank, name, key)]
    order = [name for _, name, _ in ranked]
    c_dims = result.c_dims()
    metrics = result.metrics()
    overall = result.overall
    c_scores = result.c_scores
    score_matrix = result.score_matrix

    def cell(score):
        if score is None:
            return {"score": None, "label": "", "cls": ""}
        v_label, v_class = verdict(score)
        return {"score": score, "label": v_label, "cls": v_class}

    leaderboard_rows = []
    for rank, name, key in ranked:
        leaderboard_rows.append(
            {
                "rank": rank,
                "candidate": name,
                "overall": cell(overall.get(name)),
                "c_cells": {c: cell(c_scores.get(name, {}).get(c)) for c in c_dims},
            }
        )

    matrix_rows = []
    for name in order:
        row = score_matrix.get(name, {})
        matrix_rows.append(
            {"candidate": name, "cells": {m: cell(row.get(m)) for m in metrics}}
        )

    bars = []
    for name in order:
        bars.append(
            {
                "candidate": name,
                "c_bars": [
                    {"c": c, **cell(c_scores.get(name, {}).get(c))} for c in c_dims
                ],
            }
        )

    drilldowns = [
        {"candidate": name, "html": render_html(result.reports[name], interactive=interactive, fragment=True)}
        for name in order
    ]

    tpl = _ENV.get_template("leaderboard.html.j2")
    return tpl.render(
        normalization=normalization,
        c_dims=c_dims,
        c_info={c: describe_c(c) for c in c_dims},
        metrics=metrics,
        metric_names={m: display_name(m) for m in metrics},
        leaderboard_rows=leaderboard_rows,
        matrix_rows=matrix_rows,
        bars=bars,
        drilldowns=drilldowns,
        notes=result.notes,
    )
```

- [ ] **Step 4: Implement the template** — create `src/syneva/render/html/templates/leaderboard.html.j2`:

```jinja
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>syneva benchmark leaderboard</title>
<style>
  body { font: 14px/1.4 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; color: #222; }
  h1 { border-bottom: 2px solid #444; padding-bottom: .3em; }
  h2 { margin-top: 2rem; color: #2b5; }
  table { width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: 13px; }
  th, td { border-bottom: 1px solid #eee; padding: 5px 9px; text-align: left; }
  .modeline { font-size: 12px; color: #667; background: #f4f6f8; padding: .5rem .8rem; border-radius: 6px; margin-bottom: 1rem; }
  .cell { display: inline-block; padding: 1px 8px; border-radius: 999px; font-weight: 600; }
  .v-excellent { background: #d6f5e0; color: #1a7f43; }
  .v-good { background: #e2f0d9; color: #3d7a1f; }
  .v-fair { background: #fff3cd; color: #8a6d00; }
  .v-poor { background: #fde2e2; color: #a32020; }
  .bar-track { background: #eee; border-radius: 4px; height: 12px; width: 160px; display: inline-block; vertical-align: middle; }
  .bar-fill { height: 12px; border-radius: 4px; }
  .bar-fill.v-excellent { background: #1a7f43; } .bar-fill.v-good { background: #3d7a1f; }
  .bar-fill.v-fair { background: #c9a400; } .bar-fill.v-poor { background: #a32020; }
  details.drill { margin: .6rem 0; border: 1px solid #ddd; border-radius: 6px; padding: .4rem .8rem; }
  details.drill summary { font-weight: 600; cursor: pointer; }
  .muted { color: #999; }
</style>
</head>
<body>
  <h1>syneva benchmark leaderboard</h1>
  <div class="modeline">
    Ranking normalization: <strong>{{ normalization }}</strong>.
    Displayed scores are absolute (0–1, 1 = ideal); the ranking order reflects the selected normalization.
    {% if notes %}<br>{{ notes|join("; ") }}{% endif %}
  </div>

  <h2>Leaderboard</h2>
  <table class="leaderboard">
    <thead><tr><th>#</th><th>Candidate</th><th>Overall</th>
      {% for c in c_dims %}<th title="{{ c_info[c] }}">{{ c }}</th>{% endfor %}</tr></thead>
    <tbody>
    {% for row in leaderboard_rows %}
      <tr><td>{{ row.rank }}</td><td>{{ row.candidate }}</td>
        <td>{% if row.overall.score is not none %}<span class="cell {{ row.overall.cls }}">{{ "%.3f"|format(row.overall.score) }}</span>{% else %}<span class="muted">—</span>{% endif %}</td>
        {% for c in c_dims %}{% set cc = row.c_cells[c] %}
        <td>{% if cc.score is not none %}<span class="cell {{ cc.cls }}">{{ "%.3f"|format(cc.score) }}</span>{% else %}<span class="muted">—</span>{% endif %}</td>{% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Per-dimension bars</h2>
  {% for b in bars %}
    <div><strong>{{ b.candidate }}</strong>
      {% for cb in b.c_bars %}
        <div>{{ cb.c }} <span class="bar-track"><span class="bar-fill {{ cb.cls }}" style="width: {{ (cb.score * 100)|round(1) if cb.score is not none else 0 }}%"></span></span>
          {% if cb.score is not none %}{{ "%.3f"|format(cb.score) }}{% else %}<span class="muted">—</span>{% endif %}</div>
      {% endfor %}
    </div>
  {% endfor %}

  <h2>Score matrix</h2>
  <table class="score-matrix">
    <thead><tr><th>Candidate</th>{% for m in metrics %}<th title="{{ m }}">{{ metric_names[m] }}</th>{% endfor %}</tr></thead>
    <tbody>
    {% for row in matrix_rows %}
      <tr><td>{{ row.candidate }}</td>
        {% for m in metrics %}{% set cc = row.cells[m] %}
        <td>{% if cc.score is not none %}<span class="cell {{ cc.cls }}">{{ "%.2f"|format(cc.score) }}</span>{% else %}<span class="muted">—</span>{% endif %}</td>{% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Per-candidate scorecards</h2>
  {% for d in drilldowns %}
    <details class="drill"><summary>{{ d.candidate }}</summary>
      {{ d.html|safe }}
    </details>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 5: Add `BenchmarkResult.to_html`** — in `src/syneva/benchmark/engine.py`, add this method to `BenchmarkResult` (near `to_json`):
```python
    def to_html(self, path, *, normalization: str = "absolute", interactive: bool = False) -> None:
        from pathlib import Path

        from syneva.render.html.benchmark_renderer import render_leaderboard

        Path(path).write_text(
            render_leaderboard(self, normalization=normalization, interactive=interactive)
        )
```

- [ ] **Step 6: Run, expect pass** — `uv run pytest tests/unit/render/test_leaderboard.py --no-cov -v` (4 passed).

- [ ] **Step 7: Commit** — `git commit -m "feat(render): benchmark leaderboard renderer + BenchmarkResult.to_html"`

---

## Task 3: `syneva benchmark` CLI command

**Files:** Modify `src/syneva/cli/main.py`; Test `tests/cli/test_benchmark_cmd.py`.

- [ ] **Step 1: Write the failing test** — create `tests/cli/test_benchmark_cmd.py` (mirror the subprocess style of `tests/cli/test_evaluate_cmd.py`):

```python
import json
import subprocess
import sys
from pathlib import Path

_REAL = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
_GOOD = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
_SHIFTED = Path("tests/fixtures/adult_income_syn_shifted_500.parquet").resolve()
_META = Path("tests/fixtures/metadata.json").resolve()


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "syneva.cli.main", "benchmark", *args],
        capture_output=True, text=True,
    )


def test_benchmark_writes_outputs_and_ranks_good_first(tmp_path):
    out = tmp_path / "bench"
    proc = _run([
        "--real", str(_REAL), "--metadata", str(_META),
        "--candidate", f"good={_GOOD}", "--candidate", f"shifted={_SHIFTED}",
        "--out", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert (out / "leaderboard.html").exists()
    assert (out / "benchmark.json").exists()
    loaded = json.loads((out / "benchmark.json").read_text())
    assert set(loaded["reports"]) == {"good", "shifted"}
    # stdout lists good before shifted
    assert proc.stdout.index("good") < proc.stdout.index("shifted")


def test_benchmark_malformed_candidate_exits_nonzero(tmp_path):
    proc = _run([
        "--real", str(_REAL), "--candidate", "noequalsign", "--out", str(tmp_path / "b"),
    ])
    assert proc.returncode != 0


def test_benchmark_duplicate_candidate_exits_nonzero(tmp_path):
    proc = _run([
        "--real", str(_REAL),
        "--candidate", f"a={_GOOD}", "--candidate", f"a={_SHIFTED}",
        "--out", str(tmp_path / "b"),
    ])
    assert proc.returncode != 0


def test_benchmark_unknown_normalization_exits_nonzero(tmp_path):
    proc = _run([
        "--real", str(_REAL),
        "--candidate", f"good={_GOOD}",
        "--normalization", "bogus", "--out", str(tmp_path / "b"),
    ])
    assert proc.returncode != 0
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/cli/test_benchmark_cmd.py --no-cov -v` (No such command 'benchmark').

- [ ] **Step 3: Implement** — in `src/syneva/cli/main.py`, add a `benchmark` command after the `evaluate` command (reuse `_load_df`, `_load_metadata`, `_UNSET`, the `import syneva`, and the `SynevaError` handling pattern):

```python
@app.command()
def benchmark(
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="named synthetic dataset as name=path (repeatable)"
    ),
    real: Path | None = typer.Option(None, exists=True, dir_okay=False),
    metadata: Path | None = typer.Option(None, exists=True, dir_okay=False),
    out: Path = typer.Option(Path("./syneva-benchmark")),
    holdout: Path | None = typer.Option(None, exists=True, dir_okay=False),
    preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
    tiers: str | None = typer.Option(None, help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    normalization: str = typer.Option(
        "absolute", help="ranking: absolute|linear|normal|quantile"
    ),
) -> None:
    candidates: dict[str, pd.DataFrame] = {}
    for entry in candidate:
        if "=" not in entry:
            raise typer.BadParameter(f"candidate must be name=path, got '{entry}'")
        name, _, path_str = entry.partition("=")
        if not name:
            raise typer.BadParameter(f"candidate name is empty in '{entry}'")
        if name in candidates:
            raise typer.BadParameter(f"duplicate candidate name '{name}'")
        cand_path = Path(path_str)
        if not cand_path.exists():
            raise typer.BadParameter(f"candidate '{name}' path does not exist: {path_str}")
        candidates[name] = _load_df(cand_path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        real_df = _load_df(real) if real else None
        meta = _load_metadata(metadata)
        holdout_df = _load_df(holdout) if holdout else None
        res = syneva.benchmark(
            real_df,
            candidates,
            meta,
            holdout=holdout_df,
            preset=preset,
            tiers=(tuple(t.strip() for t in tiers.split(",")) if tiers else _UNSET),
            cs=(tuple(c.strip() for c in cs.split(",")) if cs else _UNSET),
            run_utility=(True if run_utility else _UNSET),
        )
        res.to_html(out / "leaderboard.html", normalization=normalization)
        res.to_json(out / "benchmark.json")
        for rank, name, score in res.ranking(normalization=normalization):
            typer.echo(f"{rank}. {name}  {score:.4f}")
        typer.echo(f"wrote leaderboard to {out}/")
    except SynevaError as e:
        typer.echo(f"syneva error: {e}", err=True)
        raise typer.Exit(code=2) from e
```

(NOTE: `--candidate` is parsed BEFORE the try/except so `typer.BadParameter` for malformed input surfaces as typer's own exit code 2, not the SynevaError path. `--normalization bogus` reaches `res.ranking(...)`/`res.to_html(...)` inside the try → raises `SynevaError` → exit 2. Confirm `res.to_html` calls `render_leaderboard` which calls `ranking(normalization)` early; if the unknown-normalization error must surface before files are written, call `res.ranking(normalization=normalization)` once right after building `res` and before `to_html`, so a bad mode fails before writing partial output.)

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/cli/test_benchmark_cmd.py --no-cov -v` (4 passed).

- [ ] **Step 5: FULL SUITE (behavior-preservation gate)** — `uv run pytest -q`. ALL pass (2 pre-existing skips); golden UNCHANGED; coverage >= 85%. If golden fails, STOP and report BLOCKED.

- [ ] **Step 6: Commit** — `git commit -m "feat(cli): syneva benchmark command (leaderboard from N candidates)"`

---

## Self-review notes
- **Spec coverage:** render_html fragment flag + behavior preservation (T1) · render_leaderboard (ranked table + score matrix + per-C bars + collapsible drill-down) + to_html (T2) · CLI benchmark with --candidate name=path parsing + shared-config + normalization + output files (T3). Golden gate full-suite in T1 and T3.
- **Type/signature consistency:** `render_html(report, *, interactive, score_mode, fragment)`; `render_leaderboard(result, *, normalization, interactive)`; `BenchmarkResult.to_html(path, *, normalization, interactive)` mirrors `Report.to_html`; CLI reuses `_load_df`/`_load_metadata`/`_UNSET` exactly as the `evaluate` command; view dicts use `verdict()`-derived `cls`/`label`.
- **No placeholders:** complete renderer, template, and CLI code; the unknown-normalization ordering note gives a concrete guard.
- **Behavior preservation:** only the two `{% if not fragment %}` guards touch the existing template; default `fragment=False` ⇒ identical output; golden green.
