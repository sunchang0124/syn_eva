"""Streamlit UI for syneva. Run with `streamlit run` or `syn-eva ui`."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.ui import core
from syneva.utility.task import UtilityTask


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

    if run:
        if not selected:
            st.warning("Select at least one evaluator in the sidebar.")
            return
        if utility_selected and not utility_tasks:
            st.warning("A utility metric is selected but no target column was chosen.")
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
    html_str = core.report_html_str(report)
    components.html(html_str, height=900, scrolling=True)

    st.download_button("Download JSON", core.report_json_str(report), "scorecard.json")
    st.download_button("Download HTML", html_str, "scorecard.html")
    try:
        pdf = core.report_pdf_bytes(report)
        st.download_button("Download PDF", pdf, "scorecard.pdf")
    except SynevaError:
        st.caption("PDF export needs `pip install 'syneva[pdf]'`.")


if __name__ == "__main__":
    main()
