"""Streamlit UI for syneva. Run with `streamlit run` or `syn-eva ui`."""

from __future__ import annotations

import importlib.util

import streamlit as st
import streamlit.components.v1 as components

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.core.metric_info import describe_c
from syneva.ui import core
from syneva.utility.task import UtilityTask


def _sidebar() -> dict | None:
    """Render sidebar controls and return configuration dict, or None to abort."""
    with st.sidebar:
        st.header("1 · Upload data")
        real_file = st.file_uploader("Real data (CSV/Parquet)", type=["csv", "parquet"])
        syn_file = st.file_uploader("Synthetic data (CSV/Parquet)", type=["csv", "parquet"])

        if not (real_file and syn_file):
            st.info("Upload both files to continue.")
            return None

        try:
            real = core.load_table(real_file)
            synthetic = core.load_table(syn_file)
        except ValueError as e:
            st.error(str(e))
            return None

        st.header("2 · Column metadata")
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

        # Convenience selection buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select all"):
                for m in catalog:
                    st.session_state[f"chk_{m['name']}"] = True
        with col2:
            if st.button("Core only"):
                for m in catalog:
                    st.session_state[f"chk_{m['name']}"] = m["tier"] == "core"
        with col3:
            if st.button("Clear"):
                for m in catalog:
                    st.session_state[f"chk_{m['name']}"] = False

        selected: list[str] = []
        utility_selected = False
        for c in ["congruence", "coverage", "compliance", "utility"]:
            group = [m for m in catalog if m["c"] == c]
            if not group:
                continue
            st.subheader(c.capitalize(), help=describe_c(c) or None)
            for m in group:
                label = m["display_name"]
                if m["tier"] != "core":
                    label = f"{label}  ·  {m['tier']}"
                if st.checkbox(
                    label,
                    value=(m["tier"] == "core"),
                    key=f"chk_{m['name']}",
                    help=m["info"] or None,
                ):
                    selected.append(m["name"])
                    if m["c"] == "utility":
                        utility_selected = True

        st.header("4 · Utility tasks")
        utility_tasks: list[UtilityTask] | None = None
        if utility_selected:
            targets = st.multiselect("Target column(s)", options=list(real.columns))
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

    return {
        "real": real,
        "synthetic": synthetic,
        "edited": edited,
        "selected": selected,
        "utility_selected": utility_selected,
        "utility_tasks": utility_tasks,
        "run": run,
    }


def _render_results(report) -> None:
    """Render the scorecard and download buttons."""
    st.subheader("Scorecard")
    mode_label = st.radio(
        "Results display",
        ["Normalized score (0-1)", "Actual measured values"],
        horizontal=True,
        help=(
            "Normalized is easy to compare at a glance. Actual shows the raw "
            "measured statistics (KS, AUC, distances, ...) for objective reporting."
        ),
    )
    score_mode = "actual" if mode_label.startswith("Actual") else "normalized"

    html_str = core.report_html_str(report, score_mode=score_mode)
    components.html(html_str, height=900, scrolling=True)

    st.download_button("Download JSON", core.report_json_str(report), "scorecard.json")
    st.download_button("Download HTML", html_str, "scorecard.html")

    pdf_ok = importlib.util.find_spec("weasyprint") is not None
    if pdf_ok:
        try:
            st.download_button(
                "Download PDF",
                core.report_pdf_bytes(report, score_mode=score_mode),
                "scorecard.pdf",
            )
        except SynevaError:
            st.caption("PDF export unavailable (weasyprint native libraries missing).")
    else:
        st.caption("PDF export needs `pip install 'syneva[pdf]'`.")


def main() -> None:
    st.set_page_config(page_title="syneva", layout="wide")
    st.title("syneva — 7 Cs scorecard")

    cfg = _sidebar()
    if cfg and cfg["run"]:
        if not cfg["selected"]:
            st.warning("Select at least one evaluator in the sidebar.")
        elif cfg["utility_selected"] and not cfg["utility_tasks"]:
            st.warning("A utility metric is selected but no target column was chosen.")
        else:
            try:
                meta = core.metadata_from_editor(cfg["edited"])
                st.session_state["report"] = core.run_report(
                    cfg["real"], cfg["synthetic"], meta, cfg["selected"], cfg["utility_tasks"]
                )
            except (SynevaError, ValueError) as e:
                st.error(f"Evaluation failed: {e}")

    report = st.session_state.get("report")
    if report is None:
        st.write("Configure inputs in the sidebar, then click **Run evaluation**.")
        return
    _render_results(report)


if __name__ == "__main__":
    main()
