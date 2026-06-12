from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

import syneva  # noqa: F401  ensure all built-in metrics are registered
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.registry import MetricRegistry
from syneva.core.registry import registry as _global_registry
from syneva.core.report import Report
from syneva.core.runner import evaluate_with
from syneva.render.html.renderer import render_html
from syneva.utility.task import UtilityTask


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
