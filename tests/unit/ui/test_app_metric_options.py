"""The sidebar's "Advanced metric options" reach the run, and the results say so (#56)."""

import json
import subprocess
import sys
import textwrap

import pytest

# Runs the real app with uploads replaced by a small table. With "record", the two
# run functions are replaced by stand-ins that record the metric_options they were
# given and then fail, which the app reports and moves on from. With "real", the
# real run functions score the small table, so the results view renders.
_PAGE = textwrap.dedent(
    """
    import sys

    import numpy as np
    import pandas as pd
    import streamlit as st

    from syneva.core.errors import SynevaError
    from syneva.ui import app, core


    class _Upload:
        name = "candidate.csv"


    def _uploader(*args, accept_multiple_files=False, **kwargs):
        return [_Upload()] if accept_multiple_files else _Upload()


    def _table(f):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"x": rng.normal(size=60), "c": rng.choice(["a", "b"], size=60)})


    def _record(**kwargs):
        st.session_state["seen_metric_options"] = kwargs.get("metric_options")
        raise SynevaError("stand-in")


    st.file_uploader = _uploader
    core.load_table = _table
    if st.session_state.get("stand_in", True):
        core.run_report = lambda *a, **k: _record(**k)
        core.run_benchmark = lambda *a, **k: _record(**k)
    app.main()
    """
)

# Picks a mode, optionally changes that mode's option widget, presses the run
# button, and prints what the run received plus the captions on the page.
_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from streamlit.testing.v1 import AppTest

    page, mode, kind, change = sys.argv[1:5]
    at = AppTest.from_file(page)
    at.session_state["stand_in"] = kind == "record"
    at = at.run(timeout=120)
    at.radio(key="mode").set_value(mode).run(timeout=120)
    if mode == "Single scorecard":
        (clear,) = [b for b in at.button if b.label == "Clear"]
        at = clear.click().run(timeout=120)
        (profile,) = [s for s in at.selectbox if s.label == "Profile"]
        profile.set_value("fast")
        if change == "yes":
            at.checkbox(key="opt_ci_overlap_use_real_se").check()
        (run,) = [b for b in at.button if b.label == "Run evaluation"]
    else:
        (profile,) = [s for s in at.selectbox if s.key == "bm_profile"]
        profile.set_value("fast")
        if change == "yes":
            at.number_input(key="bm_opt_pmse_n_permutations").set_value(50)
        run = at.button(key="bm_run")
    at = run.click().run(timeout=120)
    assert not at.exception, at.exception
    print(json.dumps({
        "seen": at.session_state["seen_metric_options"] if kind == "record" else None,
        "captions": [c.value for c in at.caption],
    }))
    """
)

_SCORECARD = "Single scorecard"
_BENCHMARK = "Benchmark (compare datasets)"


def _drive(tmp_path, mode: str, kind: str, change: bool) -> dict:
    page = tmp_path / "page.py"
    page.write_text(_PAGE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(page), mode, kind, "yes" if change else "no"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_ticked_option_reaches_the_scorecard_run(tmp_path):
    seen = _drive(tmp_path, _SCORECARD, "record", change=True)["seen"]
    assert seen == {"ci_overlap": {"use_real_se": True}}


def test_changed_option_reaches_the_benchmark_run(tmp_path):
    seen = _drive(tmp_path, _BENCHMARK, "record", change=True)["seen"]
    assert seen == {"pmse": {"n_permutations": 50}}


@pytest.mark.parametrize("mode", [_SCORECARD, _BENCHMARK])
def test_untouched_options_send_nothing(tmp_path, mode):
    assert not _drive(tmp_path, mode, "record", change=False)["seen"]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (_SCORECARD, "Metric options: ci_overlap.use_real_se = True"),
        (_BENCHMARK, "Metric options: pmse.n_permutations = 50"),
    ],
)
def test_results_show_the_options_used(tmp_path, mode, expected):
    captions = _drive(tmp_path, mode, "real", change=True)["captions"]
    assert expected in captions


def test_results_show_no_options_caption_without_options(tmp_path):
    captions = _drive(tmp_path, _SCORECARD, "real", change=False)["captions"]
    assert not any(c.startswith("Metric options:") for c in captions)
