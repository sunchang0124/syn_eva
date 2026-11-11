import json
import subprocess
import sys
from pathlib import Path

import click
import pandas as pd

_REAL = Path("tests/fixtures/adult_income_real_500.parquet").resolve()
_SYN = Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve()
_META = Path("tests/fixtures/metadata.json").resolve()


def _evaluate(*args):
    # utf-8: typer draws its errors with box characters that cp1252 can't decode.
    return subprocess.run(
        [sys.executable, "-m", "syneva.cli.main", "evaluate", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


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


def test_load_metadata_reads_privacy_roles(tmp_path):
    from syneva.cli.main import _load_metadata

    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "columns": {
                    "zip": {"dtype": "categorical", "quasi_identifier": True},
                    "income": {"dtype": "numeric", "sensitive": True},
                }
            }
        ),
        encoding="utf-8",
    )
    meta = _load_metadata(path)
    assert meta is not None
    cols = meta.columns
    assert (cols["zip"].quasi_identifier, cols["zip"].sensitive) == (True, False)
    assert (cols["income"].quasi_identifier, cols["income"].sensitive) == (False, True)


def _evaluate_c2st(tmp_path, *seed_args):
    out = tmp_path / "rep"
    proc = _evaluate(
        "--real",
        str(_REAL),
        "--synthetic",
        str(_SYN),
        "--metadata",
        str(_META),
        "--tiers",
        "extended",
        "--cs",
        "congruence",
        "--out",
        str(out),
        *seed_args,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads((out / "scorecard.json").read_text(encoding="utf-8"))


def test_evaluate_cli_runs_at_the_given_random_state(tmp_path):
    import syneva
    from syneva.cli.main import _load_metadata

    def c2st_in_python(seed):
        rep = syneva.evaluate(
            pd.read_parquet(_REAL),
            pd.read_parquet(_SYN),
            _load_metadata(_META),
            tiers=("extended",),
            cs=("congruence",),
            random_state=seed,
        )
        (result,) = [r for r in rep.results if r.spec.name == "c2st"]
        assert result.scalars, result.skip_reason
        return result.scalars

    loaded = _evaluate_c2st(tmp_path, "--random-state", "7")
    assert loaded["run_info"]["random_state"] == 7
    (c2st,) = [r for r in loaded["results"] if r["spec"]["name"] == "c2st"]
    expected = c2st_in_python(7)
    # Only proves the seed reached c2st if seed 7 scores differently from the default.
    assert expected != c2st_in_python(42)
    assert c2st["scalars"] == expected


def test_evaluate_cli_defaults_to_random_state_42(tmp_path):
    assert _evaluate_c2st(tmp_path)["run_info"]["random_state"] == 42


def test_evaluate_cli_rejects_a_negative_random_state(tmp_path):
    proc = _evaluate(
        "--synthetic", str(_SYN), "--random-state", "-1", "--out", str(tmp_path / "rep")
    )
    # A usage error, not a numpy traceback from deep inside a metric.
    assert proc.returncode == 2, proc.stderr
    # On CI typer colours the option name piece by piece, so drop the styling first.
    stderr = click.unstyle(proc.stderr)
    assert "Traceback" not in stderr
    assert "--random-state" in stderr
