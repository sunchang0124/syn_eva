import json
import os
import subprocess
import sys
import textwrap

# The real app, with uploads replaced by a table holding a bool, a numeric, a
# categorical and a text column of numbers. RETYPE_NUMERIC names a column the
# metadata editor marks NUMERIC, as a user re-typing it would. run_report
# still runs; a wrapper records the subgroup the sidebar built and what
# subgroup_fidelity made of it.
_PAGE = textwrap.dedent(
    """
    import os

    import numpy as np
    import pandas as pd
    import streamlit as st

    from syneva.ui import app, core

    rng = np.random.default_rng(0)
    _TABLE = pd.DataFrame(
        {
            "is_female": rng.random(300) < 0.4,
            "age": rng.normal(40, 10, 300).round(),
            "job": rng.choice(["a", "b", "c"], 300),
            "code": rng.integers(10, 20, 300).astype(str),
        }
    )


    class _Upload:
        name = "data.csv"


    _run_report = core.run_report
    _metadata_rows = core.metadata_rows


    def _retyped(meta):
        rows = _metadata_rows(meta)
        for row in rows:
            if row["column"] == os.environ["RETYPE_NUMERIC"]:
                row["dtype"] = "numeric"
        return rows


    def _record(*args, **kwargs):
        report = _run_report(*args, **kwargs)
        (spec,) = kwargs["subgroup_specs"]
        (fidelity,) = [r for r in report.results if r.spec.name == "subgroup_fidelity"]
        st.session_state["seen"] = {
            "conditions": {c: repr(v) for c, v in spec.conditions.items()},
            "problem": spec.validate(args[2]),
            "scored": bool(fidelity.scalars),
            "skip_reason": fidelity.skip_reason,
        }
        return report


    st.file_uploader = lambda *a, **k: _Upload()
    core.load_table = lambda f: _TABLE.copy()
    core.run_report = _record
    if os.environ.get("RETYPE_NUMERIC"):
        core.metadata_rows = _retyped
    app.main()
    """
)

# Unticks the default metrics, ticks subgroup_fidelity, picks the columns given as
# arguments for the subgroup and prints the subgroup inputs that appear with their
# values, and any sidebar warnings. With --run it also picks True for is_female,
# runs the evaluation and prints what run_report saw.
_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from streamlit.testing.v1 import AppTest

    page, run_it, columns = sys.argv[1], sys.argv[2] == "--run", sys.argv[3:]
    at = AppTest.from_file(page).run(timeout=60)
    (clear,) = [b for b in at.button if b.label == "Clear"]
    at = clear.click().run(timeout=60)
    at = at.checkbox(key="chk_subgroup_fidelity").check().run(timeout=60)
    (picker,) = [m for m in at.multiselect if m.label == "Subgroup condition column(s)"]
    at = picker.set_value(columns).run(timeout=60)
    inputs = {
        w.key: w.value
        for w in [*at.multiselect, *at.number_input]
        if (w.key or "").startswith("sg_")
    }
    seen = None
    if run_it:
        at = at.multiselect(key="sg_vals_is_female").set_value([True]).run(timeout=60)
        (run,) = [b for b in at.button if b.label == "Run evaluation"]
        at = run.click().run(timeout=120)
        seen = at.session_state["seen"]
    assert not at.exception, at.exception
    warnings = [w.value for w in at.sidebar.warning]
    print(json.dumps({"inputs": inputs, "warnings": warnings, "seen": seen}, default=str))
    """
)


def _drive(tmp_path, *args, retype_numeric="") -> dict:
    page = tmp_path / "page.py"
    page.write_text(_PAGE, encoding="utf-8")
    env = {**os.environ, "RETYPE_NUMERIC": retype_numeric}
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(page), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_bool_subgroup_column_gets_values_and_is_scored(tmp_path):
    out = _drive(tmp_path, "--run", "is_female", "age")
    # The bool column gets a value picker; the numeric one keeps its min/max inputs.
    assert sorted(out["inputs"]) == ["sg_hi_age", "sg_lo_age", "sg_vals_is_female"]
    seen = out["seen"]
    assert seen["conditions"]["is_female"] == "[True]"
    assert seen["problem"] is None
    assert seen["scored"], seen["skip_reason"]


def test_a_column_retyped_as_numeric_gets_bounds_from_its_numbers(tmp_path):
    # Text marked NUMERIC in the editor: the builder offers a range, bounded by the
    # numbers in the text. (The run itself rejects such metadata, so none is started.)
    out = _drive(tmp_path, "--no-run", "code", retype_numeric="code")
    assert out["inputs"] == {"sg_lo_code": 10.0, "sg_hi_code": 19.0}


def test_a_numeric_column_with_no_numbers_is_flagged_not_given_made_up_bounds(tmp_path):
    # Words marked NUMERIC: there is no range to offer, so the sidebar says so and
    # adds no condition for the column, instead of a (0, 0) range that matches nothing.
    out = _drive(tmp_path, "--no-run", "job", retype_numeric="job")
    assert out["inputs"] == {}
    assert any("'job'" in w and "no numbers" in w for w in out["warnings"])
