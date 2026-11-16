# tests/unit/congruence/test_correlation_diff.py
import numpy as np
import pandas as pd
import pytest

from syneva.congruence.correlation_diff import CorrelationDifference
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

NUM = ColumnType.NUMERIC
CAT = ColumnType.CATEGORICAL


def _meta_of(**types: ColumnType) -> Metadata:
    return Metadata(columns={n: ColumnMetadata(name=n, dtype=t) for n, t in types.items()})


def test_identical_corr_matrix_score_perfect():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"a": a, "b": b, "c": ["X", "Y"] * 100})
    result = CorrelationDifference().compute(df, df, _meta_of(a=NUM, b=NUM, c=CAT))
    assert result.scalars is not None
    assert result.scalars["score"] >= 0.99


def test_destroyed_correlation_lowers_score():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.05, size=200)
    real = pd.DataFrame({"a": a, "b": b, "c": ["X"] * 200})
    syn = pd.DataFrame({"a": a, "b": rng.permutation(b), "c": ["X"] * 200})
    result = CorrelationDifference().compute(real, syn, _meta_of(a=NUM, b=NUM, c=CAT))
    assert result.scalars is not None
    assert result.scalars["score"] < 0.7


def _pair_with_pearson(rho: float) -> pd.DataFrame:
    """Two columns whose Pearson correlation is exactly ``rho``.

    ``x`` and ``z`` have zero mean, equal norm and are orthogonal, so
    ``b = rho * x + sqrt(1 - rho**2) * z`` correlates with ``x`` at exactly ``rho``.
    """
    x = np.tile([1.0, 1.0, -1.0, -1.0], 25)
    z = np.tile([1.0, -1.0, 1.0, -1.0], 25)
    return pd.DataFrame({"a": x, "b": rho * x + np.sqrt(1 - rho**2) * z})


def test_pearson_sign_flip_counts_full_difference():
    result = CorrelationDifference().compute(
        _pair_with_pearson(0.4), _pair_with_pearson(-0.4), _meta_of(a=NUM, b=NUM)
    )
    assert result.per_column is not None
    assert result.per_column["a|b"]["abs_corr_diff"] == pytest.approx(0.8)


def test_pearson_difference_is_capped_at_one():
    result = CorrelationDifference().compute(
        _pair_with_pearson(0.9), _pair_with_pearson(-0.9), _meta_of(a=NUM, b=NUM)
    )
    assert result.per_column is not None and result.scalars is not None
    assert result.per_column["a|b"]["abs_corr_diff"] == pytest.approx(1.0)
    assert result.scalars["score"] == pytest.approx(0.0)


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
    result = CorrelationDifference().compute(
        _pair_with_eta(0.6), _pair_with_eta(0.8), _meta_of(a=NUM, c=CAT)
    )
    assert result.per_column is not None
    # η: 0.8 - 0.6 = 0.2  (η² would give 0.64 - 0.36 = 0.28)
    assert result.per_column["a|c"]["abs_corr_diff"] == pytest.approx(0.2)
