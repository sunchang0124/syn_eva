import json
import subprocess
import sys
import textwrap

# Runs the real app with uploads replaced by a small table and the two run
# functions replaced by stand-ins that record the seed they were given. The
# stand-ins then fail, which the app reports and moves on from, so nothing is
# rendered from a fake result.
_PAGE = textwrap.dedent(
    """
    import pandas as pd
    import streamlit as st

    from syneva.core.errors import SynevaError
    from syneva.ui import app, core


    class _Upload:
        name = "candidate.csv"


    def _uploader(*args, accept_multiple_files=False, **kwargs):
        return [_Upload()] if accept_multiple_files else _Upload()


    def _record(**kwargs):
        st.session_state["seen_random_state"] = kwargs["random_state"]
        raise SynevaError("stand-in")


    st.file_uploader = _uploader
    core.load_table = lambda f: pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    core.run_report = lambda *a, **k: _record(**k)
    core.run_benchmark = lambda *a, **k: _record(**k)
    app.main()
    """
)

# Picks a mode, sets the seed to 7, presses that mode's run button and prints
# the seed the run function received.
_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from streamlit.testing.v1 import AppTest

    page, mode = sys.argv[1], sys.argv[2]
    at = AppTest.from_file(page).run(timeout=60)
    at.radio(key="mode").set_value(mode).run(timeout=60)
    at.number_input(key="random_state").set_value(7)
    if mode == "Single scorecard":
        # Untick the default metrics first: a ticked utility metric asks for a
        # target column before anything runs.
        (clear,) = [b for b in at.button if b.label == "Clear"]
        at = clear.click().run(timeout=60)
        (profile,) = [s for s in at.selectbox if s.label == "Profile"]
        profile.set_value("fast")
        (run,) = [b for b in at.button if b.label == "Run evaluation"]
    else:
        run = at.button(key="bm_run")
    at = run.click().run(timeout=60)
    assert not at.exception, at.exception
    print(json.dumps(at.session_state["seen_random_state"]))
    """
)


def _seed_seen_by(tmp_path, mode: str) -> int:
    page = tmp_path / "page.py"
    page.write_text(_PAGE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(page), mode],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_scorecard_runs_at_the_seed_chosen_in_the_sidebar(tmp_path):
    assert _seed_seen_by(tmp_path, "Single scorecard") == 7


def test_benchmark_runs_at_the_seed_chosen_in_the_sidebar(tmp_path):
    assert _seed_seen_by(tmp_path, "Benchmark (compare datasets)") == 7
