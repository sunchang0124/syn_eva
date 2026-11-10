import subprocess
import sys
import textwrap

# A Streamlit page that renders the benchmark view for two hand-built candidates:
# dcr failed for A and scored 0.2 for B, so it is left out of the comparison.
_PAGE = textwrap.dedent(
    """
    from syneva.benchmark.engine import BenchmarkResult
    from syneva.core.errors import MetricError
    from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
    from syneva.core.metric import MetricResult, MetricSpec
    from syneva.core.report import Report
    from syneva.core.run_info import RunInfo
    from syneva.ui.app import _render_benchmark

    meta = Metadata(columns={"x": ColumnMetadata(name="x", dtype=ColumnType.NUMERIC)})
    info = RunInfo(random_state=42, started_at=0.0, finished_at=1.0, library_versions={})


    def spec(name, c):
        return MetricSpec(
            name=name,
            c=c,
            tier="core",
            data_types=frozenset(["static"]),
            requires_real=True,
            scope="table-level",
        )


    def report(dcr):
        ks = MetricResult(spec=spec("ks_statistic", "congruence"), scalars={"score": 0.7})
        if dcr is None:
            d = MetricResult(spec=spec("dcr", "compliance"), error=MetricError("x"))
        else:
            d = MetricResult(spec=spec("dcr", "compliance"), scalars={"score": dcr})
        return Report(metadata=meta, results=[ks, d], run_info=info)


    _render_benchmark(BenchmarkResult(reports={"A": report(None), "B": report(0.2)}), "absolute")
    """
)

# Runs the page through Streamlit's AppTest in a subprocess, which keeps
# Streamlit's native-library side effects out of the test process (see
# test_app_importable.py), and prints the text of every element: the embedded
# leaderboard is an iframe whose HTML is its srcdoc.
_DRIVER = textwrap.dedent(
    """
    import sys
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(sys.argv[1]).run(timeout=60)
    assert not at.exception, at.exception
    for frame in at.get("iframe"):
        print("IFRAME:", frame.proto.srcdoc)
    for caption in at.caption:
        print("CAPTION:", caption.value)
    for markdown in at.markdown:
        print("MARKDOWN:", markdown.value)
    """
)


def test_benchmark_view_shows_metrics_left_out_of_the_comparison_once(tmp_path):
    page = tmp_path / "page.py"
    page.write_text(_PAGE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(page)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    # Shown by the embedded leaderboard, which escapes names, and nowhere else.
    iframe = proc.stdout.partition("\nCAPTION:")[0]
    iframe = iframe.partition("\nMARKDOWN:")[0]
    assert iframe.startswith("IFRAME:")
    assert iframe.count("dcr not compared: failed for A") == 1
    assert proc.stdout.count("dcr not compared: failed for A") == 1
