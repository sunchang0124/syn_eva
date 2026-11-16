# tests/unit/congruence/test_correlation_diff.py
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from syneva.congruence.correlation_diff import CorrelationDifference
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


def _meta():
    return Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.NUMERIC),
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
        }
    )


def test_identical_corr_matrix_score_perfect():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"a": a, "b": b, "c": ["X", "Y"] * 100})
    r = CorrelationDifference().compute(df, df, _meta())
    assert r.scalars["score"] >= 0.99


def test_destroyed_correlation_lowers_score():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.05, size=200)
    real = pd.DataFrame({"a": a, "b": b, "c": ["X"] * 200})
    syn = pd.DataFrame({"a": a, "b": rng.permutation(b), "c": ["X"] * 200})
    r = CorrelationDifference().compute(real, syn, _meta())
    assert r.scalars["score"] < 0.7


def _numeric_meta():
    return Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.NUMERIC),
        }
    )


def _pair_with_pearson(r: float) -> pd.DataFrame:
    """Two columns whose Pearson correlation is exactly ``r``.

    ``x`` and ``z`` have zero mean, equal norm and are orthogonal, so
    ``b = r * x + sqrt(1 - r**2) * z`` correlates with ``x`` at exactly ``r``.
    """
    x = np.tile([1.0, 1.0, -1.0, -1.0], 25)
    z = np.tile([1.0, -1.0, 1.0, -1.0], 25)
    return pd.DataFrame({"a": x, "b": r * x + np.sqrt(1 - r**2) * z})


def test_pearson_sign_flip_counts_full_difference():
    r = CorrelationDifference().compute(
        _pair_with_pearson(0.4), _pair_with_pearson(-0.4), _numeric_meta()
    )
    assert r.per_column is not None and r.scalars is not None
    assert r.per_column["a|b"]["abs_corr_diff"] == pytest.approx(0.8)


def test_pearson_difference_is_capped_at_one():
    r = CorrelationDifference().compute(
        _pair_with_pearson(0.9), _pair_with_pearson(-0.9), _numeric_meta()
    )
    assert r.per_column is not None and r.scalars is not None
    assert r.per_column["a|b"]["abs_corr_diff"] == pytest.approx(1.0)
    assert r.scalars["score"] == pytest.approx(0.0)


def _pair_with_eta(eta: float) -> pd.DataFrame:
    """A numeric column whose correlation ratio on ``c`` is exactly ``eta``.

    Group means sit at ``±m`` with within-group deviations of ``±1``, so the
    between-group share of the variance is ``m**2 / (m**2 + 1) = eta**2``.
    """
    m = eta / np.sqrt(1 - eta**2)
    a = np.tile([m + 1, m - 1, -m + 1, -m - 1], 25)
    c = np.tile(["X", "X", "Y", "Y"], 25)
    return pd.DataFrame({"a": a, "c": c})


def test_numeric_categorical_pair_compares_eta_not_eta_squared():
    meta = Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
        }
    )
    r = CorrelationDifference().compute(_pair_with_eta(0.6), _pair_with_eta(0.8), meta)
    assert r.per_column is not None and r.scalars is not None
    # η: 0.8 - 0.6 = 0.2  (η² would give 0.64 - 0.36 = 0.28)
    assert r.per_column["a|c"]["abs_corr_diff"] == pytest.approx(0.2)


def _mixed_frame(seed: int, rho: float, shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=120)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=120)
    c = np.where(x + shift * rng.normal(size=120) > 0, "P", "Q")
    d = rng.choice(["u", "v", "w"], size=120)
    return pd.DataFrame({"a": x, "b": y, "c": c, "d": d})


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    rho_real=st.floats(min_value=-0.99, max_value=0.99),
    rho_syn=st.floats(min_value=-0.99, max_value=0.99),
    shift=st.floats(min_value=0.0, max_value=3.0),
)
def test_every_difference_and_the_mean_lie_in_unit_interval(seed, rho_real, rho_syn, shift):
    meta = Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.NUMERIC),
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
            "d": ColumnMetadata(name="d", dtype=ColumnType.CATEGORICAL),
        }
    )
    r = CorrelationDifference().compute(
        _mixed_frame(seed, rho_real, shift), _mixed_frame(seed + 1, rho_syn, shift), meta
    )
    assert r.per_column is not None and r.scalars is not None
    for pair in r.per_column.values():
        assert 0.0 <= pair["abs_corr_diff"] <= 1.0
    mean = r.scalars["mean_abs_corr_diff"]
    assert 0.0 <= mean <= 1.0
    # The bound means the score never has to be clipped.
    assert r.scalars["score"] == pytest.approx(1.0 - mean)
