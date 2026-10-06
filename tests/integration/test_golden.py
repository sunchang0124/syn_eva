import json
import math
from pathlib import Path

import pytest

import syneva

GOLDEN = Path(__file__).parent.parent / "golden" / "adult_income_good.scorecard.json"

# Float leaves may differ in the last few ULPs across platforms / BLAS builds
# (summation order), so they are compared with a relative tolerance.
REL_TOL = 1e-9


def _scrub(d: dict) -> dict:
    """Strip volatile fields before diffing."""
    d = json.loads(json.dumps(d))
    d["run_info"]["started_at"] = 0
    d["run_info"]["finished_at"] = 0
    d["run_info"]["library_versions"] = {k: "PINNED" for k in d["run_info"]["library_versions"]}
    return d


def _first_mismatch(actual, expected, path: str = "") -> str | None:
    """Return the path of the first difference, or None if the trees match.

    Floats compare with ``REL_TOL``; everything else (keys, lengths, types,
    strings, ints, bools, None) must match exactly.
    """
    if isinstance(actual, float) or isinstance(expected, float):
        both_numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in (actual, expected)
        )
        if not both_numeric:
            return f"{path}: {actual!r} != {expected!r}"
        if math.isnan(actual) and math.isnan(expected):
            return None
        if math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=0.0):
            return None
        return f"{path}: {actual!r} != {expected!r}"
    if type(actual) is not type(expected):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return f"{path}: keys differ {sorted(set(actual) ^ set(expected))}"
        for k in actual:
            if (m := _first_mismatch(actual[k], expected[k], f"{path}/{k}")) is not None:
                return m
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            if (m := _first_mismatch(a, e, f"{path}[{i}]")) is not None:
                return m
        return None
    return None if actual == expected else f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"x": 0.014511895104386774}, {"x": 0.014511895104386663}),
        ({"x": [0.00035936146304506897]}, {"x": [0.0003593614630450688]}),
        ({"x": float("nan")}, {"x": float("nan")}),
        ({"x": 1.0}, {"x": 1}),
        ({"x": "a", "y": None, "z": True}, {"x": "a", "y": None, "z": True}),
    ],
)
def test_first_mismatch_accepts_equivalent_trees(actual, expected):
    assert _first_mismatch(actual, expected) is None


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({"x": 0.1234}, {"x": 0.1235}),
        ({"x": 1e-12}, {"x": 2e-12}),
        ({"x": 0.0}, {"x": 1e-300}),
        ({"x": float("nan")}, {"x": 0.5}),
        ({"x": 1.0}, {"x": "1.0"}),
        ({"x": 1.0}, {"x": None}),
        ({"x": 1.0}, {"x": True}),
        ({"x": 1}, {"x": 2}),
        ({"x": "a"}, {"x": "b"}),
        ({"x": 1.0}, {"y": 1.0}),
        ({"x": [1.0]}, {"x": [1.0, 2.0]}),
    ],
)
def test_first_mismatch_rejects_real_differences(actual, expected):
    assert _first_mismatch(actual, expected) is not None


def test_adult_income_good_matches_golden(real_df, syn_good_df, metadata, update_golden):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core",), run_utility=False)
    actual = _scrub(rep.to_dict())
    if update_golden or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True))
        return
    expected = _scrub(json.loads(GOLDEN.read_text()))
    mismatch = _first_mismatch(actual, expected)
    assert mismatch is None, (
        f"Golden mismatch at {mismatch}. If intentional, run "
        "`uv run pytest tests/integration/test_golden.py --update-golden`"
    )
