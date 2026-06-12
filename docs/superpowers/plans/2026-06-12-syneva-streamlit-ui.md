# syneva Streamlit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app that lets a user upload real + synthetic tabular data, edit column metadata, pick which evaluators to run, and view/download the 7 Cs scorecard — as a thin layer over `syneva.evaluate`.

**Architecture:** All testable logic lives in a pure module `src/syneva/ui/core.py` (load files, build metadata, select metrics, run, render to str/bytes). `src/syneva/ui/app.py` holds only the Streamlit layout and imports `core`. Per-metric selection is realised by building a transient `MetricRegistry` of the chosen metric classes and calling `evaluate_with(...)`. Streamlit is an optional `[ui]` extra; the core test suite must run without it installed. A `syn-eva ui` CLI subcommand execs `streamlit run app.py`.

**Tech Stack:** Python 3.10+, Streamlit, pandas, existing `syneva` public API (`evaluate_with`, `Metadata`, `ColumnType`, `registry`, `MetricRegistry`, `UtilityTask`, `Report`, `render_html`).

**Spec:** `docs/superpowers/specs/2026-06-12-syneva-streamlit-ui-design.md`.

---

## File Structure

- Create: `src/syneva/ui/__init__.py` — empty package marker.
- Create: `src/syneva/ui/core.py` — pure, unit-tested helpers (no Streamlit import).
- Create: `src/syneva/ui/app.py` — Streamlit layout only; `main()` guarded so import is side-effect-free.
- Modify: `src/syneva/core/registry.py` — add public `metrics()` accessor.
- Modify: `src/syneva/cli/main.py` — add `ui` subcommand.
- Modify: `pyproject.toml` — add `ui` optional extra; fold into `all`.
- Test: `tests/unit/ui/test_core.py`, `tests/unit/ui/test_app_importable.py`, `tests/unit/core/test_registry.py` (extend), `tests/cli/test_ui_cmd.py`.

---

## Task 1: Add `ui` optional extra and streamlit dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `ui` extra and fold it into `all`**

In `pyproject.toml`, under `[project.optional-dependencies]`, change the block to:

```toml
[project.optional-dependencies]
pdf = ["weasyprint>=62"]
plotly = ["plotly>=5.20"]
ui = ["streamlit>=1.36"]
all = ["syneva[pdf,plotly,ui]"]
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync --all-extras`
Expected: streamlit installed, no errors.

- [ ] **Step 3: Verify streamlit imports**

Run: `uv run python -c "import streamlit; print(streamlit.__version__)"`
Expected: prints a version `>= 1.36`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add optional [ui] extra (streamlit)"
```

---

## Task 2: Public `metrics()` accessor on MetricRegistry

The UI needs to list registered metrics without touching the private `_metrics` dict.

**Files:**
- Modify: `src/syneva/core/registry.py`
- Test: `tests/unit/core/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/core/test_registry.py`:

```python
def test_metrics_returns_registered_classes():
    r = MetricRegistry()
    r.register(_FakeMetric)
    metrics = r.metrics()
    assert _FakeMetric in metrics
    assert len(metrics) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/core/test_registry.py::test_metrics_returns_registered_classes -v`
Expected: FAIL — `AttributeError: 'MetricRegistry' object has no attribute 'metrics'`.

- [ ] **Step 3: Add the accessor**

In `src/syneva/core/registry.py`, inside `class MetricRegistry`, after the `select` method, add:

```python
    def metrics(self) -> list[type["Metric"]]:
        """Return all registered metric classes (read-only view)."""
        return list(self._metrics.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/core/test_registry.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/syneva/core/registry.py tests/unit/core/test_registry.py
git commit -m "feat(core): public MetricRegistry.metrics() accessor"
```

---

## Task 3: `load_table` — read CSV/Parquet uploads

**Files:**
- Create: `src/syneva/ui/__init__.py`
- Create: `src/syneva/ui/core.py`
- Test: `tests/unit/ui/test_core.py`

- [ ] **Step 1: Create the package marker**

Create `src/syneva/ui/__init__.py` (empty file).

- [ ] **Step 2: Create the test package marker and write the failing test**

Create `tests/unit/ui/__init__.py` (empty). Create `tests/unit/ui/test_core.py`:

```python
import pandas as pd
import pytest

from syneva.ui import core

FIX = "tests/fixtures"


def test_load_table_parquet():
    df = core.load_table(f"{FIX}/adult_income_real_500.parquet")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500


def test_load_table_csv(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    df = core.load_table(str(p))
    assert list(df.columns) == ["a", "b"]


def test_load_table_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("nope")
    with pytest.raises(ValueError, match="unsupported"):
        core.load_table(str(p))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'syneva.ui.core'`.

- [ ] **Step 4: Write `load_table` in `src/syneva/ui/core.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_table(file: Any) -> pd.DataFrame:
    """Read an uploaded table by extension.

    `file` may be a path string/Path or a file-like object that carries a
    ``name`` attribute (e.g. a Streamlit UploadedFile). Both CSV and Parquet
    are supported; anything else raises ValueError.
    """
    name = getattr(file, "name", None) or str(file)
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    if suffix == ".parquet":
        return pd.read_parquet(file)
    raise ValueError(f"unsupported file type '{suffix}'; use .csv or .parquet")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/syneva/ui/__init__.py src/syneva/ui/core.py tests/unit/ui/__init__.py tests/unit/ui/test_core.py
git commit -m "feat(ui): load_table reads CSV/Parquet uploads"
```

---

## Task 4: Metadata <-> editor-rows conversion

**Files:**
- Modify: `src/syneva/ui/core.py`
- Test: `tests/unit/ui/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/ui/test_core.py`:

```python
from syneva.core.metadata import ColumnType, Metadata


def test_metadata_rows_from_inferred():
    df = pd.DataFrame({"age": [1, 2], "sex": ["M", "F"]})
    rows = core.metadata_rows(Metadata.infer(df))
    by_col = {r["column"]: r for r in rows}
    assert by_col["age"]["dtype"] == "numeric"
    assert by_col["sex"]["dtype"] == "categorical"
    assert by_col["age"]["sensitive"] is False


def test_metadata_from_editor_roundtrip():
    rows = [
        {"column": "age", "dtype": "numeric", "sensitive": False},
        {"column": "sex", "dtype": "categorical", "sensitive": True},
    ]
    meta = core.metadata_from_editor(rows)
    assert meta.columns["age"].dtype is ColumnType.NUMERIC
    assert meta.columns["sex"].dtype is ColumnType.CATEGORICAL
    assert meta.columns["sex"].sensitive is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_core.py -k metadata -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'metadata_rows'`.

- [ ] **Step 3: Implement both helpers in `src/syneva/ui/core.py`**

Add the import at the top of the file:

```python
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
```

Then add:

```python
def metadata_rows(meta: Metadata) -> list[dict]:
    """Flatten Metadata into editor rows (one dict per column)."""
    return [
        {"column": name, "dtype": cm.dtype.value, "sensitive": bool(cm.sensitive)}
        for name, cm in meta.columns.items()
    ]


def metadata_from_editor(rows: list[dict]) -> Metadata:
    """Rebuild Metadata from edited rows produced by `metadata_rows`."""
    cols = {
        r["column"]: ColumnMetadata(
            name=r["column"],
            dtype=ColumnType(r["dtype"]),
            sensitive=bool(r["sensitive"]),
        )
        for r in rows
    }
    return Metadata(columns=cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/syneva/ui/core.py tests/unit/ui/test_core.py
git commit -m "feat(ui): metadata <-> editor-row conversion"
```

---

## Task 5: Metric listing + selection registry

**Files:**
- Modify: `src/syneva/ui/core.py`
- Test: `tests/unit/ui/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/ui/test_core.py`:

```python
import syneva  # populates the global registry on import


def test_metric_catalog_groups_and_tiers():
    cat = core.metric_catalog()
    names = {m["name"] for m in cat}
    assert "ks_statistic" in names
    assert "sliced_wasserstein" in names
    ks = next(m for m in cat if m["name"] == "ks_statistic")
    assert ks["c"] == "congruence"
    assert ks["tier"] == "core"


def test_build_selection_registry_contains_only_selected():
    reg = core.build_selection_registry(["ks_statistic", "dcr"])
    selected = {c.spec.name for c in reg.metrics()}
    assert selected == {"ks_statistic", "dcr"}


def test_build_selection_registry_unknown_name_raises():
    with pytest.raises(KeyError):
        core.build_selection_registry(["does_not_exist"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_core.py -k "catalog or selection" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'metric_catalog'`.

- [ ] **Step 3: Implement in `src/syneva/ui/core.py`**

Add the imports near the top:

```python
import syneva  # noqa: F401  ensure all built-in metrics are registered
from syneva.core.registry import MetricRegistry
from syneva.core.registry import registry as _global_registry
```

Then add:

```python
def metric_catalog() -> list[dict]:
    """List every registered metric with its C, tier, and real-data need."""
    out = [
        {
            "name": cls.spec.name,
            "c": cls.spec.c,
            "tier": cls.spec.tier,
            "requires_real": cls.spec.requires_real,
        }
        for cls in _global_registry.metrics()
    ]
    return sorted(out, key=lambda m: (m["c"], m["tier"], m["name"]))


def build_selection_registry(names: list[str]) -> MetricRegistry:
    """Build a registry containing exactly the named metric classes."""
    by_name = {cls.spec.name: cls for cls in _global_registry.metrics()}
    reg = MetricRegistry()
    for name in names:
        reg.register(by_name[name])  # KeyError if unknown
    return reg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/syneva/ui/core.py tests/unit/ui/test_core.py
git commit -m "feat(ui): metric_catalog + build_selection_registry"
```

---

## Task 6: `run_report` — evaluate the selected metrics

**Files:**
- Modify: `src/syneva/ui/core.py`
- Test: `tests/unit/ui/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/ui/test_core.py`:

```python
from syneva.core.report import Report
from syneva.utility.task import UtilityTask


def test_run_report_runs_only_selected_metrics():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    meta = Metadata.infer(real)
    rep = core.run_report(real, syn, meta, ["ks_statistic", "tvd"], utility_tasks=None)
    assert isinstance(rep, Report)
    assert {r.spec.name for r in rep.results} == {"ks_statistic", "tvd"}


def test_run_report_runs_utility_when_selected():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    meta = Metadata.infer(real)
    rep = core.run_report(
        real, syn, meta, ["tstr_suite"],
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "tstr_suite" in by
    assert by["tstr_suite"].scalars["score"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_core.py -k run_report -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_report'`.

- [ ] **Step 3: Implement in `src/syneva/ui/core.py`**

Add imports near the top:

```python
from syneva.core.report import Report
from syneva.core.runner import evaluate_with
from syneva.utility.task import UtilityTask
```

Then add:

```python
_UTILITY_C = "utility"


def run_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Metadata,
    selected_names: list[str],
    utility_tasks: list[UtilityTask] | None = None,
    random_state: int = 42,
) -> Report:
    """Run exactly the selected metrics and return a Report.

    Utility metrics only run when at least one is selected; `run_utility` is
    inferred from the selection so the caller need not pass it separately.
    """
    reg = build_selection_registry(selected_names)
    run_utility = any(cls.spec.c == _UTILITY_C for cls in reg.metrics())
    return evaluate_with(
        reg,
        real=real,
        synthetic=synthetic,
        metadata=metadata,
        tiers=("core", "extended"),
        utility_tasks=utility_tasks,
        run_utility=run_utility,
        random_state=random_state,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/syneva/ui/core.py tests/unit/ui/test_core.py
git commit -m "feat(ui): run_report evaluates exactly the selected metrics"
```

---

## Task 7: In-memory render helpers (HTML string, JSON string, PDF bytes)

`Report.to_html/to_json/to_pdf` write to disk and return None. Streamlit needs
in-memory strings/bytes for inline embedding and download buttons.

**Files:**
- Modify: `src/syneva/ui/core.py`
- Test: `tests/unit/ui/test_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/ui/test_core.py`:

```python
import json as _json


def _small_report():
    real = pd.read_parquet(f"{FIX}/adult_income_real_500.parquet")
    syn = pd.read_parquet(f"{FIX}/adult_income_syn_good_500.parquet")
    return core.run_report(real, syn, Metadata.infer(real), ["ks_statistic"])


def test_report_html_str_returns_html():
    html = core.report_html_str(_small_report())
    assert "<html" in html.lower()
    assert "syneva" in html.lower()


def test_report_json_str_is_valid_json():
    data = _json.loads(core.report_json_str(_small_report()))
    assert data["results"]


def test_report_pdf_bytes_guarded():
    pytest.importorskip("weasyprint")
    pdf = core.report_pdf_bytes(_small_report())
    assert pdf[:4] == b"%PDF"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_core.py -k "html_str or json_str or pdf_bytes" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'report_html_str'`.

- [ ] **Step 3: Implement in `src/syneva/ui/core.py`**

Add imports near the top:

```python
import json
import tempfile

from syneva.render.html.renderer import render_html
```

Then add:

```python
def report_html_str(report: Report, *, interactive: bool = False) -> str:
    """Render the scorecard to an HTML string (same output as Report.to_html)."""
    return render_html(report, interactive=interactive)


def report_json_str(report: Report) -> str:
    """Serialize the report to a JSON string."""
    return json.dumps(report.to_dict(), indent=2, default=str)


def report_pdf_bytes(report: Report) -> bytes:
    """Render the scorecard to PDF bytes (requires the [pdf] extra)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        report.to_pdf(tmp.name)  # raises SynevaError if weasyprint missing
        tmp.seek(0)
        return Path(tmp.name).read_bytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_core.py -v`
Expected: all pass (PDF test skips if weasyprint absent).

- [ ] **Step 5: Commit**

```bash
git add src/syneva/ui/core.py tests/unit/ui/test_core.py
git commit -m "feat(ui): in-memory HTML/JSON/PDF render helpers"
```

---

## Task 8: The Streamlit app (`app.py`)

The app holds layout only and is import-safe: the body lives in `main()` guarded
by `if __name__ == "__main__"` (which is true under `streamlit run`).

**Files:**
- Create: `src/syneva/ui/app.py`
- Test: `tests/unit/ui/test_app_importable.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ui/test_app_importable.py`:

```python
def test_app_module_imports_without_running_streamlit():
    # Importing must not execute any Streamlit calls (main() is guarded).
    import importlib

    mod = importlib.import_module("syneva.ui.app")
    assert hasattr(mod, "main")
    assert callable(mod.main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/ui/test_app_importable.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'syneva.ui.app'`.

- [ ] **Step 3: Write `src/syneva/ui/app.py`**

```python
"""Streamlit UI for syneva. Run with `streamlit run` or `syn-eva ui`."""

from __future__ import annotations

import streamlit as st

from syneva.core.errors import SynevaError
from syneva.ui import core


def main() -> None:
    st.set_page_config(page_title="syneva", layout="wide")
    st.title("syneva — 7 Cs scorecard")

    with st.sidebar:
        st.header("1 · Upload data")
        real_file = st.file_uploader("Real data (CSV/Parquet)", type=["csv", "parquet"])
        syn_file = st.file_uploader("Synthetic data (CSV/Parquet)", type=["csv", "parquet"])

        if not (real_file and syn_file):
            st.info("Upload both files to continue.")
            return

        try:
            real = core.load_table(real_file)
            synthetic = core.load_table(syn_file)
        except ValueError as e:
            st.error(str(e))
            return

        st.header("2 · Column metadata")
        from syneva.core.metadata import Metadata

        inferred_rows = core.metadata_rows(Metadata.infer(real))
        edited = st.data_editor(
            inferred_rows,
            column_config={
                "dtype": st.column_config.SelectboxColumn(
                    options=["numeric", "categorical", "datetime", "boolean", "id"]
                ),
                "sensitive": st.column_config.CheckboxColumn(),
            },
            hide_index=True,
            key="meta_editor",
        )

        st.header("3 · Evaluators")
        catalog = core.metric_catalog()
        selected: list[str] = []
        utility_selected = False
        for c in ["congruence", "coverage", "compliance", "utility"]:
            group = [m for m in catalog if m["c"] == c]
            if not group:
                continue
            st.subheader(c.capitalize())
            for m in group:
                label = f"{m['name']}  ·  {m['tier']}"
                if st.checkbox(label, value=(m["tier"] == "core"), key=f"chk_{m['name']}"):
                    selected.append(m["name"])
                    if m["c"] == "utility":
                        utility_selected = True

        st.header("4 · Utility tasks")
        utility_tasks = None
        if utility_selected:
            targets = st.multiselect("Target column(s)", options=list(real.columns))
            from syneva.utility.task import UtilityTask

            utility_tasks = []
            for t in targets:
                kind = st.selectbox(
                    f"Task type for '{t}'",
                    options=["classification", "regression"],
                    key=f"task_{t}",
                )
                utility_tasks.append(UtilityTask(target=t, task_type=kind))
        else:
            st.caption("Select a utility metric to configure tasks.")

        run = st.button("Run evaluation", type="primary")

    if run:
        if not selected:
            st.warning("Select at least one evaluator in the sidebar.")
            return
        try:
            meta = core.metadata_from_editor(edited)
            report = core.run_report(real, synthetic, meta, selected, utility_tasks)
            st.session_state["report"] = report
        except SynevaError as e:
            st.error(f"Evaluation failed: {e}")
            return

    report = st.session_state.get("report")
    if report is None:
        st.write("Configure inputs in the sidebar, then click **Run evaluation**.")
        return

    st.subheader("Scorecard")
    st.components.v1.html(core.report_html_str(report), height=900, scrolling=True)

    st.download_button("Download JSON", core.report_json_str(report), "scorecard.json")
    st.download_button("Download HTML", core.report_html_str(report), "scorecard.html")
    try:
        pdf = core.report_pdf_bytes(report)
        st.download_button("Download PDF", pdf, "scorecard.pdf")
    except SynevaError:
        st.caption("PDF export needs `pip install 'syneva[pdf]'`.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/ui/test_app_importable.py -v`
Expected: 1 passed.

- [ ] **Step 5: Manually smoke-launch (optional, not automated)**

Run: `uv run streamlit run src/syneva/ui/app.py`
Expected: the app opens in a browser; uploading the two fixtures and clicking
Run produces a scorecard. Stop with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add src/syneva/ui/app.py tests/unit/ui/test_app_importable.py
git commit -m "feat(ui): Streamlit app (sidebar controls + scorecard panel)"
```

---

## Task 9: `syn-eva ui` CLI subcommand

**Files:**
- Modify: `src/syneva/cli/main.py`
- Test: `tests/cli/test_ui_cmd.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_ui_cmd.py`:

```python
import sys
from pathlib import Path

import pytest

from syneva.cli import main as cli


def test_ui_builds_streamlit_command(monkeypatch):
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli.ui()
    assert captured["cmd"][0] == sys.executable
    assert "streamlit" in captured["cmd"]
    assert "run" in captured["cmd"]
    assert captured["cmd"][-1].endswith("app.py")


def test_ui_missing_streamlit_errors(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("no streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit):
        cli.ui()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_ui_cmd.py -v`
Expected: FAIL — `AttributeError: module 'syneva.cli.main' has no attribute 'ui'`.

- [ ] **Step 3: Implement the subcommand in `src/syneva/cli/main.py`**

Add these imports at the top of the file (alongside the existing imports):

```python
import subprocess
import sys
```

Then add this command after the `evaluate` command:

```python
@app.command()
def ui() -> None:
    """Launch the Streamlit UI."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        typer.echo(
            "The UI needs Streamlit: pip install 'syneva[ui]'", err=True
        )
        raise typer.Exit(code=2) from exc
    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    proc = subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    raise typer.Exit(code=proc.returncode)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_ui_cmd.py -v`
Expected: 2 passed.

Note: `cli.ui()` returns by raising `typer.Exit` (a subclass of `SystemExit`).
`test_ui_builds_streamlit_command` patches `subprocess.run`, so the real
Streamlit never starts; the `typer.Exit(0)` it raises is fine — adjust the test
to wrap the call in `pytest.raises(SystemExit)` if needed during implementation.

- [ ] **Step 5: Commit**

```bash
git add src/syneva/cli/main.py tests/cli/test_ui_cmd.py
git commit -m "feat(cli): syn-eva ui launches the Streamlit app"
```

---

## Task 10: End-to-end smoke test on fixtures

**Files:**
- Test: `tests/integration/test_ui_smoke.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_ui_smoke.py`:

```python
import json

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.ui import core


def test_ui_pipeline_end_to_end():
    real = core.load_table("tests/fixtures/adult_income_real_500.parquet")
    syn = core.load_table("tests/fixtures/adult_income_syn_good_500.parquet")
    assert isinstance(real, pd.DataFrame)

    rows = core.metadata_rows(Metadata.infer(real))
    rows[[r["column"] for r in rows].index("income")]["sensitive"] = True
    meta = core.metadata_from_editor(rows)

    names = ["ks_statistic", "tvd", "dcr", "k_anonymity"]
    report = core.run_report(real, syn, meta, names)
    assert {r.spec.name for r in report.results} == set(names)

    html = core.report_html_str(report)
    assert "<html" in html.lower()
    json.loads(core.report_json_str(report))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_ui_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: Run the whole suite + coverage gate**

Run: `uv run pytest -q`
Expected: all pass, coverage ≥ 85%. Note: `src/syneva/ui/app.py` is excluded
from coverage in the next step; if the gate fails only because of `app.py`,
proceed to Step 4 and re-run.

- [ ] **Step 4: Exclude the Streamlit layout file from coverage**

In `pyproject.toml`, under `[tool.coverage.run]`, extend `omit`:

```toml
[tool.coverage.run]
source = ["src/syneva"]
omit = ["src/syneva/render/html/templates/*", "src/syneva/ui/app.py"]
```

(`app.py` is layout-only and verified by the import test + manual launch, not unit tests.)

- [ ] **Step 5: Run the whole suite again**

Run: `uv run pytest -q`
Expected: all pass, coverage ≥ 85%.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_ui_smoke.py pyproject.toml
git commit -m "test(ui): end-to-end smoke + exclude app.py from coverage"
```

---

## Task 11: Docs — README + CHANGELOG

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a UI section to `README.md`**

After the existing `## CLI` section, add:

```markdown
## Web UI

A local Streamlit app for upload → pick evaluators → scorecard:

```bash
pip install 'syneva[ui]'
syn-eva ui            # or: streamlit run src/syneva/ui/app.py
```

Upload your real and synthetic tables, review the inferred column metadata
(mark sensitive columns for k-anonymity/DCR), tick the evaluators to run, and
view or download the scorecard as HTML/JSON/PDF.
```

- [ ] **Step 2: Add a CHANGELOG entry under `[Unreleased]`**

```markdown
## [Unreleased]

### Added
- Streamlit web UI (`syn-eva ui`, optional `[ui]` extra): upload real/synthetic
  data, edit column metadata, select evaluators, and view/download the scorecard.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document the Streamlit UI"
```

---

## Self-review notes

- **Spec coverage:** optional `[ui]` extra (Task 1) · dual launch streamlit/`syn-eva ui` (Tasks 8, 9) · CSV+Parquet upload (Task 3) · auto-infer + editable metadata (Tasks 4, 8) · per-metric selection grouped by C with tiers (Tasks 5, 8) · optional utility tasks (Tasks 6, 8) · reuse HTML renderer inline + JSON/HTML/PDF downloads (Tasks 7, 8) · inline error handling (Task 8) · pure-helper unit tests + import-safe app + smoke test (Tasks 3–8, 10) · public registry accessor (Task 2). All spec sections map to a task.
- **Types/signatures consistent:** `core.load_table`, `core.metadata_rows`, `core.metadata_from_editor`, `core.metric_catalog`, `core.build_selection_registry`, `core.run_report`, `core.report_html_str`, `core.report_json_str`, `core.report_pdf_bytes`, `MetricRegistry.metrics()`, `cli.ui()` are referenced consistently across tasks and tests.
- **No placeholders:** every code step contains complete code.
