# syneva Streamlit UI — Design

**Date:** 2026-06-12
**Status:** Approved (pending spec review)
**Depends on:** `syneva` v0.1 public API (`evaluate`, `Metadata`, `ColumnType`, `registry`, `UtilityTask`, `Report`).

## Goal

A local, single-user web UI that lets a user upload real and synthetic tabular
data, review/edit column metadata, choose which evaluators to run, and view the
resulting 7 Cs scorecard — all without writing code. The UI is a thin
presentation layer over the existing `syneva.evaluate()` engine.

## Scope

**In scope (v1):**
- Local tool launched on the user's own machine (no auth, no hosting, no concurrency).
- Upload real + synthetic data as CSV or Parquet.
- Auto-inferred, user-editable column metadata (dtype + sensitive flag).
- Per-metric evaluator selection, grouped by C, with tier badges.
- Optional utility-task configuration (target column + task type).
- Scorecard display (reusing the existing HTML renderer) + JSON/HTML/PDF download.

**Out of scope (v1):**
- Multi-user / hosted / public deployment, authentication, rate limiting.
- Longitudinal/relational data (the core library does not support these yet).
- Editing or authoring custom metrics from the UI.
- Persisting sessions/history across restarts.

## Architecture

- New module: `src/syneva/ui/app.py` — a Streamlit application.
- New optional extra in `pyproject.toml`: `ui = ["streamlit>=1.36"]`; folded into
  `all`. Streamlit is **not** a core dependency.
- Launch paths:
  1. `uv run streamlit run src/syneva/ui/app.py`
  2. `syn-eva ui` — a new CLI subcommand that locates `app.py` and execs
     `streamlit run` on it. If Streamlit is not installed, it prints an
     install hint (`pip install 'syneva[ui]'`) and exits non-zero.
- The app imports only the `syneva` public API. The only possible core addition
  is a small read-only helper to list registered metrics grouped by C/tier
  (see "Library touch-points").

## Layout (Streamlit sidebar + main panel)

**Sidebar — controls, top to bottom:**
1. **Upload:** two `st.file_uploader` widgets — real, then synthetic. Accept
   `.csv` and `.parquet`.
2. **Metadata editor:** once both files load, infer metadata via
   `Metadata.infer(real)` and render an `st.data_editor` table with one row per
   column: editable `dtype` (dropdown over `ColumnType` values) and a boolean
   `sensitive` checkbox.
3. **Evaluators:** metrics read live from `registry`, grouped under Congruence /
   Coverage / Compliance / Utility, each with a tier badge (core/extended) and a
   checkbox. Convenience controls: "Select all", "Core only", "Clear".
4. **Utility tasks:** shown only when at least one `utility` metric is ticked. A
   multiselect of candidate target columns and a per-target classification/
   regression choice, defaulting to `suggest_tasks(metadata)`.
5. **Run evaluation** button.

**Main panel:**
- Before a run: brief instructions / empty state.
- After a run: the scorecard (per-C summary scores, then per-metric tables and
  plots), followed by download buttons for JSON, HTML, and PDF.

## Data flow

On **Run evaluation**:
1. Read both uploads into `pandas.DataFrame` (`_load_table` dispatches on suffix).
2. Build `Metadata` from the edited table (`_metadata_from_editor`).
3. Resolve ticked metric names, plus `utility_tasks` / `run_utility` from the
   utility controls.
4. Run the evaluation. Because `evaluate()` selects by `tiers`/`cs` rather than
   individual metric names, arbitrary per-metric selection is realised by
   building a transient `MetricRegistry` containing only the selected metric
   classes (looked up from the global `registry`) and calling
   `evaluate_with(that_registry, real=..., synthetic=..., tiers=("core","extended"),
   utility_tasks=..., run_utility=...)`. This yields a `Report` containing
   exactly the chosen metrics. `run_utility` is set True automatically when any
   utility metric is selected.
5. Store the `Report` in `st.session_state` so widget tweaks don't force a
   re-run; the user re-runs explicitly via the button.

## Results rendering

Reuse the existing HTML renderer: embed `report.to_html()` output via
`st.components.v1.html(...)` so the in-app view is identical to the downloadable
HTML file. Download buttons:
- **JSON** — `report.to_json` bytes.
- **HTML** — `report.to_html` string.
- **PDF** — `report.to_pdf`; the button is disabled with a tooltip/hint when
  `weasyprint` is not importable.

## Error handling

All user-facing errors are surfaced inline with `st.error`/`st.warning`, never a
raw stack trace:
- Unsupported file extension → clear message naming accepted formats.
- Real/synthetic schema mismatch → catch `SchemaError`, list extra/missing columns.
- No evaluators selected → warn and block the run.
- Utility metric selected but no target chosen → warn.
- Any `SynevaError` from `evaluate()` → shown as an error banner.
Per-metric failures are already isolated by the runner and appear as error cards
within the scorecard rather than aborting the whole run.

## Library touch-points

- Possibly add `syneva.list_metrics()` (or similar) returning, for each
  registered metric, `(name, c, tier, requires_real)` — a thin read-only view
  over `registry` so the UI need not reach into private internals. If trivial to
  do via the existing public `registry`, no core change is required.
- The UI uses `evaluate_with` (already in `syneva.core.runner`) for the
  transient-registry run; optionally re-export it from `syneva` for convenience.
- Add the `syn-eva ui` subcommand in `src/syneva/cli/main.py`.
- No changes to metric implementations or `evaluate()` semantics.

## Testing

- Pure helpers are unit-tested without a running Streamlit server:
  - `_load_table(path_or_buffer)` — CSV and Parquet, plus unsupported suffix.
  - `_metadata_from_editor(rows)` — round-trips an edited table into `Metadata`,
    including dtype override and sensitive flag.
  - `_selected_metric_classes(...)` — maps checkbox state to the list of
    selected metric classes and whether utility is among them.
  - `_build_selection_registry(classes)` — returns a `MetricRegistry`
    containing exactly those classes.
- A smoke test imports `syneva.ui.app` and drives the helpers end-to-end on the
  `adult_income` fixtures, asserting a `Report` is produced.
- Streamlit widget rendering itself is not unit-tested (standard practice); the
  `[ui]` extra and Streamlit imports are guarded so the core test suite runs
  without Streamlit installed.

## Open questions

None outstanding; revisit hosting/auth only if the tool later needs to be shared.
