import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


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
    loaded = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert loaded["results"]


def test_evaluate_cli_accepts_holdout(tmp_path):
    real = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
    syn = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
    meta = Path("tests/fixtures/metadata.json").resolve()
    holdout = tmp_path / "holdout.csv"
    pd.read_parquet(real).sample(frac=0.3, random_state=1).to_csv(holdout, index=False)
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
            "--holdout",
            str(holdout),
            "--tiers",
            "core,extended",
            "--cs",
            "compliance",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    by = {r["spec"]["name"]: r for r in loaded["results"]}
    assert by["mia_auc"]["scalars"] is not None
    assert "mia_auc" in by["mia_auc"]["scalars"]


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


def test_load_metadata_reads_utf8(tmp_path, cp1252_locale):
    from syneva.cli.main import _load_metadata

    path = tmp_path / "metadata.json"
    raw = {"columns": {"年龄": {"dtype": "numeric"}, "≥50K": {"dtype": "categorical"}}}
    path.write_bytes(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    meta = _load_metadata(path)
    assert meta is not None
    assert set(meta.columns) == {"年龄", "≥50K"}
