"""`--metric-option metric.option=value` on `syn-eva evaluate` and `benchmark` (#55).

The library calls are swapped for recorders, so these tests see exactly what the
CLI passes on without running any metric. The recorder then raises, which the CLI
reports as a syneva error.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import syneva
from syneva.cli.main import app
from syneva.core.errors import SynevaError

_REAL = str(Path("tests/fixtures/adult_income_real_500.parquet").resolve())
_SYN = str(Path("tests/fixtures/adult_income_syn_good_500.parquet").resolve())


@pytest.fixture
def seen(monkeypatch):
    calls: list[dict] = []

    def _record(*args, **kwargs):
        calls.append(kwargs)
        raise SynevaError("stand-in")

    monkeypatch.setattr(syneva, "evaluate", _record)
    monkeypatch.setattr(syneva, "benchmark", _record)
    return calls


def _flat(output: str) -> str:
    """The error text without typer's box, so a message wrapped across lines matches."""
    return " ".join(output.replace("│", " ").split())


def _evaluate(tmp_path, *options):
    args = ["evaluate", "--real", _REAL, "--synthetic", _SYN, "--out", str(tmp_path)]
    for o in options:
        args += ["--metric-option", o]
    return CliRunner().invoke(app, args)


def _benchmark(tmp_path, *options):
    args = ["benchmark", "--real", _REAL, "--candidate", f"good={_SYN}", "--out", str(tmp_path)]
    for o in options:
        args += ["--metric-option", o]
    return CliRunner().invoke(app, args)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("false", False),
        ("5", 5),
        ("1.5", 1.5),
        ('"x"', "x"),
        ("abc", "abc"),
        ("[1, 2]", [1, 2]),
    ],
)
def test_value_is_read_as_json_or_kept_as_text(tmp_path, seen, raw, expected):
    # An unregistered metric, so no known option type applies; evaluate() rejects it.
    _evaluate(tmp_path, f"custom.opt={raw}")
    value = seen[0]["metric_options"]["custom"]["opt"]
    assert value == expected
    assert type(value) is type(expected)


def test_flags_combine_into_one_mapping(tmp_path, seen):
    _evaluate(
        tmp_path,
        "ci_overlap.use_real_se=true",
        "pmse.n_permutations=50",
        "pmse.other=1",
    )
    assert seen[0]["metric_options"] == {
        "ci_overlap": {"use_real_se": True},
        "pmse": {"n_permutations": 50, "other": 1},
    }


def test_value_may_contain_dots_and_equals(tmp_path, seen):
    _evaluate(tmp_path, "m.opt=a.b=c")
    assert seen[0]["metric_options"] == {"m": {"opt": "a.b=c"}}


def test_without_the_flag_no_metric_options_are_passed(tmp_path, seen):
    _evaluate(tmp_path)
    assert seen[0].get("metric_options") is None


def test_benchmark_passes_metric_options(tmp_path, seen):
    _benchmark(tmp_path, "pmse.n_permutations=5")
    assert seen[0]["metric_options"] == {"pmse": {"n_permutations": 5}}


@pytest.mark.parametrize(
    "entry", ["ci_overlap.use_real_se", "use_real_se=true", ".use_real_se=true", "ci_overlap.=true"]
)
@pytest.mark.parametrize("run", [_evaluate, _benchmark])
def test_malformed_entry_is_rejected_before_running(tmp_path, seen, run, entry):
    result = run(tmp_path, entry)
    assert result.exit_code == 2
    assert entry in _flat(result.output)
    assert seen == []


@pytest.mark.parametrize("run", [_evaluate, _benchmark])
def test_repeated_option_is_rejected(tmp_path, seen, run):
    result = run(tmp_path, "pmse.n_permutations=5", "pmse.n_permutations=10")
    assert result.exit_code == 2
    assert "pmse.n_permutations" in _flat(result.output)
    assert seen == []


@pytest.mark.parametrize(
    ("entry", "expects"),
    [
        ("ci_overlap.use_real_se=False", "true or false"),
        ("ci_overlap.use_real_se=yes", "true or false"),
        ("ci_overlap.use_real_se=", "true or false"),
        ("pmse.n_permutations=ten", "a whole number"),
    ],
)
def test_text_for_a_bool_or_number_option_is_rejected(tmp_path, seen, entry, expects):
    # Kept as text, "False" or "yes" would be a non-empty string, which a metric
    # reads as true: the opposite of what was asked.
    result = _evaluate(tmp_path, entry)
    assert result.exit_code == 2
    assert expects in _flat(result.output)
    assert seen == []


def test_quoted_text_is_still_passed_on_as_text(tmp_path, seen):
    _evaluate(tmp_path, 'ci_overlap.use_real_se="False"')
    assert seen[0]["metric_options"] == {"ci_overlap": {"use_real_se": "False"}}


def test_spaces_around_names_and_value_are_ignored(tmp_path, seen):
    _evaluate(tmp_path, " ci_overlap . use_real_se = true ")
    assert seen[0]["metric_options"] == {"ci_overlap": {"use_real_se": True}}
