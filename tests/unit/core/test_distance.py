import numpy as np
import pandas as pd
import pytest

from syneva.core.distance import gower_matrix, gower_nearest_index, numeric_ranges
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "num": ColumnMetadata(name="num", dtype=ColumnType.NUMERIC),
            "cat": ColumnMetadata(name="cat", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_rows_zero_distance():
    df = pd.DataFrame({"num": [0.0, 5.0, 10.0], "cat": ["x", "y", "x"]})
    d = gower_matrix(df, df, _meta())
    assert np.allclose(np.diag(d), 0.0)


def test_mixed_example_values():
    a = pd.DataFrame({"num": [0.0], "cat": ["x"]})
    b = pd.DataFrame({"num": [10.0, 5.0], "cat": ["y", "x"]})
    d = gower_matrix(a, b, _meta())
    assert d.shape == (1, 2)
    assert abs(d[0, 0] - 1.0) < 1e-9
    assert abs(d[0, 1] - 0.25) < 1e-9


def test_no_usable_columns_zero():
    df = pd.DataFrame({"id": ["a", "b"]})
    meta = Metadata(columns={"id": ColumnMetadata(name="id", dtype=ColumnType.ID)})
    d = gower_matrix(df, df, meta)
    assert d.shape == (2, 2)
    assert np.allclose(d, 0.0)


def test_gower_nan_numeric_contributes_zero():
    import numpy as np

    a = pd.DataFrame({"num": [float("nan")], "cat": ["x"]})
    b = pd.DataFrame({"num": [5.0], "cat": ["x"]})
    d = gower_matrix(a, b, _meta())
    # numeric pair has NaN -> 0 contribution; cat equal -> 0; mean 0
    assert np.isclose(d[0, 0], 0.0)


def test_gower_constant_numeric_column_zero_contribution():
    import numpy as np

    meta = Metadata(columns={"num": ColumnMetadata(name="num", dtype=ColumnType.NUMERIC)})
    df = pd.DataFrame({"num": [7.0, 7.0, 7.0]})
    d = gower_matrix(df, df, meta)
    assert np.allclose(d, 0.0)


def _num_meta():
    return Metadata(columns={"num": ColumnMetadata(name="num", dtype=ColumnType.NUMERIC)})


def test_explicit_ranges_fix_the_scale():
    a = pd.DataFrame({"num": [0.0]})
    b = pd.DataFrame({"num": [2.0, 100.0]})
    d = gower_matrix(a, b, _num_meta(), ranges={"num": 4.0})
    # 2/4 inside the range; 100/4 is clipped to 1.
    assert np.allclose(d, [[0.5, 1.0]])


def test_explicit_zero_range_is_zero_if_equal_else_one():
    a = pd.DataFrame({"num": [7.0]})
    b = pd.DataFrame({"num": [7.0, 7.5, float("nan")]})
    d = gower_matrix(a, b, _num_meta(), ranges={"num": 0.0})
    assert np.allclose(d, [[0.0, 1.0, 0.0]])


def test_numeric_ranges_from_finite_values_only():
    df = pd.DataFrame({"num": [1.0, 4.0, float("nan"), float("inf")], "cat": ["x", "y", "x", "y"]})
    assert numeric_ranges(df, _meta()) == {"num": 3.0}


def test_numeric_ranges_skips_columns_without_finite_values():
    df = pd.DataFrame({"num": [float("nan"), float("nan")]})
    assert numeric_ranges(df, _num_meta()) == {}


def test_column_missing_from_ranges_is_skipped():
    a = pd.DataFrame({"num": [0.0], "cat": ["x"]})
    b = pd.DataFrame({"num": [5.0], "cat": ["y"]})
    d = gower_matrix(a, b, _meta(), ranges={})
    assert np.allclose(d, [[1.0]])


def test_gower_nearest_index_worked_example():
    real = pd.DataFrame({"num": [0.0, 10.0], "cat": ["x", "y"]})
    syn = pd.DataFrame({"num": [10.0, 0.0, 5.0], "cat": ["y", "x", "x"]})
    # Row 0 (0, x): distances 1.0, 0.0, 0.25 -> 1. Row 1 (10, y): 0.0, 1.0, 0.75 -> 0.
    idx = gower_nearest_index(real, syn, _meta(), {"num": 10.0}, block_size=1)
    assert idx.tolist() == [1, 0]


def test_gower_nearest_index_ties_go_to_first_synthetic_row():
    real = pd.DataFrame({"num": [5.0], "cat": ["x"]})
    syn = pd.DataFrame({"num": [10.0, 0.0, 10.0], "cat": ["x", "x", "x"]})
    # All three synthetic rows sit 0.25 away.
    assert gower_nearest_index(real, syn, _meta(), {"num": 10.0}).tolist() == [0]


@pytest.mark.parametrize("block_size", [1, 7, 10, 40, 1000])
def test_gower_nearest_index_matches_full_matrix_argmin(block_size):
    rng = np.random.default_rng(0)
    real = pd.DataFrame(
        {"num": rng.integers(0, 5, 40).astype(float), "cat": rng.choice(list("ab"), 40)}
    )
    syn = pd.DataFrame(
        {"num": rng.integers(0, 5, 25).astype(float), "cat": rng.choice(list("ab"), 25)}
    )
    ranges = numeric_ranges(real, _meta())
    expected = np.argmin(gower_matrix(real, syn, _meta(), ranges), axis=1)
    idx = gower_nearest_index(real, syn, _meta(), ranges, block_size=block_size)
    assert idx.tolist() == expected.tolist()


def test_gower_nearest_index_rejects_non_positive_block_size():
    df = pd.DataFrame({"num": [1.0], "cat": ["x"]})
    with pytest.raises(ValueError, match="block_size"):
        gower_nearest_index(df, df, _meta(), {"num": 1.0}, block_size=0)
