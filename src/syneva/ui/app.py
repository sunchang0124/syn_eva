"""Streamlit UI for syneva. Run with `streamlit run` or `syn-eva ui`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from syneva.core.errors import SynevaError
from syneva.core.metadata import Metadata
from syneva.core.metric_info import describe_c
from syneva.ui import core, reload_notice
from syneva.utility.task import UtilityTask


def _sidebar() -> dict | None:
    """Render sidebar controls and return configuration dict, or None to abort."""
    with st.sidebar:
        st.header("1 · Upload data")
        real_file = st.file_uploader("Real data (CSV/Parquet)", type=["csv", "parquet"])
        syn_file = st.file_uploader("Synthetic data (CSV/Parquet)", type=["csv", "parquet"])
        holdout_file = st.file_uploader(
            "Holdout / test data (optional, CSV/Parquet)", type=["csv", "parquet"]
        )

        if not (real_file and syn_file):
            st.info("Upload both files to continue.")
            return None

        try:
            real = core.load_table(real_file)
            synthetic = core.load_table(syn_file)
        except ValueError as e:
            st.error(str(e))
            return None

        holdout = None
        if holdout_file is not None:
            try:
                holdout = core.load_table(holdout_file)
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

        profile = st.selectbox(
            "Profile",
            ["Custom", "fast", "full", "privacy"],
            index=0,
            help="A preset runs a curated metric set; Custom lets you pick metrics by hand.",
        )

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
        for c in ["congruence", "coverage", "compliance", "utility", "fairness", "preservation"]:
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
                    disabled=profile != "Custom",
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
                task_type = "classification" if kind == "classification" else "regression"
                utility_tasks.append(UtilityTask(target=t, task_type=task_type))
        else:
            st.caption("Select a utility metric to configure tasks.")

        st.header("5 · Fairness")
        fairness_selected = any(m["c"] == "fairness" for m in catalog if m["name"] in selected)
        fairness_specs = None
        if fairness_selected:
            from syneva import FairnessSpec

            cols = list(real.columns)
            protected = st.selectbox("Protected attribute", options=cols)
            outcome = st.selectbox("Outcome column", options=cols, index=min(1, len(cols) - 1))
            if protected and outcome and protected != outcome:
                fairness_specs = [FairnessSpec(protected_attribute=protected, outcome=outcome)]
            elif protected == outcome:
                st.warning("Protected attribute and outcome must be different columns.")
        else:
            st.caption("Select the statistical-parity metric to configure fairness.")

        st.header("6 · Preservation")
        preservation_subgroup_selected = any(
            m["c"] == "preservation"
            and m["name"] in selected
            and m["name"] in ("subgroup_fidelity", "minority_utility_gap", "minority_privacy_risk")
            for m in catalog
        )
        subgroup_specs = None
        if preservation_subgroup_selected:
            from syneva import SubgroupSpec

            sg_name = st.text_input("Subgroup name", value="subgroup 1")
            sg_cols = st.multiselect("Subgroup condition column(s)", options=list(real.columns))
            conditions: dict = {}
            for col in sg_cols:
                if pd.api.types.is_numeric_dtype(real[col]):
                    lo = st.number_input(
                        f"'{col}' min", value=float(real[col].min()), key=f"sg_lo_{col}"
                    )
                    hi = st.number_input(
                        f"'{col}' max", value=float(real[col].max()), key=f"sg_hi_{col}"
                    )
                    conditions[col] = (lo, hi)
                else:
                    vals = st.multiselect(
                        f"'{col}' values",
                        options=sorted(real[col].dropna().unique().tolist()),
                        key=f"sg_vals_{col}",
                    )
                    if vals:
                        conditions[col] = vals
            if conditions:
                subgroup_specs = [SubgroupSpec(name=sg_name, conditions=conditions)]
            else:
                st.info("No subgroup defined — only the automatic preservation metrics will run.")
        else:
            st.caption("Select a subgroup metric to configure subgroups.")

        run = st.button("Run evaluation", type="primary")

    return {
        "real": real,
        "synthetic": synthetic,
        "edited": edited,
        "selected": selected,
        "utility_selected": utility_selected,
        "utility_tasks": utility_tasks,
        "fairness_selected": fairness_selected,
        "fairness_specs": fairness_specs,
        "subgroup_specs": subgroup_specs,
        "holdout": holdout,
        "preset": None if profile == "Custom" else profile,
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
                        subgroup_specs=cfg["subgroup_specs"],
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
            "Upload real data and synthetic candidates in the sidebar, "
            "then click **Run benchmark**."
        )
        return
    result, normalization = bm
    _render_benchmark(result, normalization)


if __name__ == "__main__":
    changed_at = reload_notice.begin()
    main()
    reload_notice.end(changed_at)
