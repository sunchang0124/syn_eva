import json
import subprocess
import sys
from pathlib import Path

_REAL = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
_GOOD = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
_SHIFTED = Path("tests/fixtures/adult_income_syn_shifted_500.parquet").resolve()
_META = Path("tests/fixtures/metadata.json").resolve()


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "syneva.cli.main", "benchmark", *args],
        capture_output=True,
        text=True,
    )


def test_benchmark_writes_outputs_and_ranks_good_first(tmp_path):
    out = tmp_path / "bench"
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--metadata",
            str(_META),
            "--candidate",
            f"good={_GOOD}",
            "--candidate",
            f"shifted={_SHIFTED}",
            "--out",
            str(out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "leaderboard.html").exists()
    assert (out / "benchmark.json").exists()
    loaded = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert set(loaded["reports"]) == {"good", "shifted"}
    assert proc.stdout.index("good") < proc.stdout.index("shifted")


def test_benchmark_malformed_candidate_exits_nonzero(tmp_path):
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--candidate",
            "noequalsign",
            "--out",
            str(tmp_path / "b"),
        ]
    )
    assert proc.returncode != 0


def test_benchmark_duplicate_candidate_exits_nonzero(tmp_path):
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--candidate",
            f"a={_GOOD}",
            "--candidate",
            f"a={_SHIFTED}",
            "--out",
            str(tmp_path / "b"),
        ]
    )
    assert proc.returncode != 0


def test_benchmark_unknown_normalization_exits_nonzero(tmp_path):
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--candidate",
            f"good={_GOOD}",
            "--normalization",
            "bogus",
            "--out",
            str(tmp_path / "b"),
        ]
    )
    assert proc.returncode != 0


def test_benchmark_prints_metrics_left_out_of_the_comparison(tmp_path):
    import pandas as pd

    good = pd.read_parquet(_GOOD)
    # One row repeated: correlation_difference is skipped for this candidate only.
    constant = tmp_path / "constant.parquet"
    pd.concat([good.head(1)] * 50, ignore_index=True).to_parquet(constant)
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--metadata",
            str(_META),
            "--candidate",
            f"good={_GOOD}",
            "--candidate",
            f"constant={constant}",
            "--out",
            str(tmp_path / "bench"),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "correlation_difference not compared: skipped for constant" in proc.stdout


def _benchmark_random_state(tmp_path, *seed_args):
    out = tmp_path / "bench"
    proc = _run(
        [
            "--real",
            str(_REAL),
            "--metadata",
            str(_META),
            "--candidate",
            f"good={_GOOD}",
            "--preset",
            "fast",
            "--out",
            str(out),
            *seed_args,
        ]
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    return loaded["random_state"], loaded["reports"]["good"]["run_info"]["random_state"]


def test_benchmark_runs_at_the_given_random_state(tmp_path):
    assert _benchmark_random_state(tmp_path, "--random-state", "7") == (7, 7)


def test_benchmark_defaults_to_random_state_42(tmp_path):
    assert _benchmark_random_state(tmp_path) == (42, 42)
