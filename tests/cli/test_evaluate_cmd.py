import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_cli_writes_outputs(tmp_path):
    real = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
    syn = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
    meta = Path("tests/fixtures/metadata.json").resolve()
    out = tmp_path / "report"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "syneva.cli.main",
            "evaluate",
            "--real",
            str(real),
            "--synthetic",
            str(syn),
            "--metadata",
            str(meta),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "scorecard.json").exists()
    assert (out / "scorecard.html").exists()
    loaded = json.loads((out / "scorecard.json").read_text())
    assert loaded["results"]


def test_evaluate_cli_handles_missing_real(tmp_path):
    syn = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "syneva.cli.main",
            "evaluate",
            "--synthetic",
            str(syn),
            # Restrict to congruence: every congruence metric requires real data,
            # so with no --real the selection is empty and evaluate() raises.
            # (k_anonymity is synth-only, so an unrestricted run would succeed.)
            "--cs",
            "congruence",
            "--out",
            str(tmp_path / "r"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "no runnable" in proc.stderr.lower() or "syneva" in proc.stderr.lower()


def test_cli_preset_full_runs(tmp_path):
    real = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
    syn = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
    out = tmp_path / "rep"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "syneva.cli.main",
            "evaluate",
            "--real",
            str(real),
            "--synthetic",
            str(syn),
            "--preset",
            "full",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "scorecard.json").exists()


def test_cli_unknown_preset_exits_nonzero(tmp_path):
    syn = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
    out = tmp_path / "rep"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "syneva.cli.main",
            "evaluate",
            "--synthetic",
            str(syn),
            "--preset",
            "bogus",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
