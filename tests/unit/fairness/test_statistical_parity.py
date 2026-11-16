import pandas as pd
import pytest

from syneva import FairnessSpec
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.fairness.statistical_parity import StatisticalParity


def _meta_gy():
    return Metadata(
        columns={
            "g": ColumnMetadata(name="g", dtype=ColumnType.CATEGORICAL),
            "y": ColumnMetadata(name="y", dtype=ColumnType.CATEGORICAL),
        }
    )


def _biased(seed_a_pos, seed_b_pos):
    y = [1] * seed_a_pos + [0] * (50 - seed_a_pos) + [1] * seed_b_pos + [0] * (50 - seed_b_pos)
    return pd.DataFrame({"g": ["A"] * 50 + ["B"] * 50, "y": y})


def test_preserved_bias_low_drift():
    real = _biased(40, 10)
    syn = _biased(40, 10)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] < 0.05
    assert r.scalars["score"] > 0.95


def test_destroyed_bias_high_drift():
    real = _biased(40, 10)
    syn = _biased(25, 25)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars["parity_drift"] > 0.5
    assert r.scalars["score"] < 0.5


def test_no_specs_is_skipped():
    df = _biased(40, 10)
    r = StatisticalParity(specs=[]).compute(df, df, _meta_gy())
    assert r.scalars is None
    assert r.skip_reason


def _groups(**rates):
    """Frame with 100 rows per group; ``rates`` maps group -> count of favorable rows."""
    parts = [
        pd.DataFrame({"g": [g] * 100, "y": [1] * pos + [0] * (100 - pos)})
        for g, pos in rates.items()
    ]
    return pd.concat(parts, ignore_index=True)


def test_erased_group_is_maximal_drift():
    real = _groups(A=60, B=30)
    syn = _groups(A=60)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars is not None
    assert r.scalars["parity_drift"] == 1.0
    assert r.scalars["score"] == 0.0
    assert any("'B'" in n and "missing" in n for n in r.notes)


def test_erased_group_with_matching_gap_is_not_perfect():
    # The gap is 0.30 in both frames, but group C was erased and A drifted.
    real = _groups(A=50, B=50, C=20)
    syn = _groups(A=80, B=50)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars is not None
    assert r.scalars["parity_drift"] == 1.0
    assert r.scalars["score"] == 0.0


def test_synthetic_only_group_is_ignored():
    real = _groups(A=60, B=30)
    syn = _groups(A=60, B=30, Z=0)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars is not None
    assert r.scalars["spd_synthetic"] == pytest.approx(0.3)
    assert r.scalars["parity_drift"] < 1e-9
    assert r.scalars["score"] > 0.999


def test_group_with_no_real_outcomes_is_not_required():
    # Group C has rows in real but every outcome is null, so it is not a real group.
    real = pd.concat(
        [_groups(A=60, B=30), pd.DataFrame({"g": ["C"] * 10, "y": [None] * 10})],
        ignore_index=True,
    )
    syn = _groups(A=60, B=30)
    r = StatisticalParity(specs=[FairnessSpec("g", "y")]).compute(real, syn, _meta_gy())
    assert r.scalars is not None
    assert r.scalars["parity_drift"] < 1e-9
    assert not any("missing" in n for n in r.notes)


def test_erased_group_counts_as_one_spec_in_the_mean():
    real = _groups(A=60, B=30)
    real["h"] = ["X", "Y"] * 100  # X and Y have equal rates in every g block
    syn = real[real["g"] == "A"]  # g group B erased; h groups X and Y both kept
    r = StatisticalParity(specs=[FairnessSpec("g", "y"), FairnessSpec("h", "y")]).compute(
        real, syn, _meta_gy()
    )
    assert r.scalars is not None and r.per_column is not None
    assert r.per_column["g|y"]["parity_drift"] == 1.0
    assert r.per_column["h|y"]["parity_drift"] == pytest.approx(0.0)
    assert r.scalars["score"] == pytest.approx(0.5)
