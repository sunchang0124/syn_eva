"""Columns that are constant in the real data have no spread to standardise by (#39)."""

import numpy as np
import pandas as pd
import pytest

from syneva.congruence.dimension_wise_means import DimensionWiseMeans
from syneva.congruence.quantile_mse import QuantileMSE
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

METRICS = [QuantileMSE, DimensionWiseMeans]
N = 500


def _numeric_meta(*names: str) -> Metadata:
    return Metadata(columns={n: ColumnMetadata(name=n, dtype=ColumnType.NUMERIC) for n in names})


def _frames(k_syn: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real: a normal column ``x`` and a column ``k`` that is always 5.0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    real = pd.DataFrame({"x": x, "k": np.full(N, 5.0)})
    syn = pd.DataFrame({"x": x + rng.normal(scale=0.1, size=N), "k": k_syn})
    return real, syn


SYNTHETIC_K = {
    "noise": 5.0 + np.random.default_rng(1).normal(scale=0.01, size=N),
    "exact match": np.full(N, 5.0),
    "other constant": np.full(N, 6.0),
}


@pytest.mark.parametrize("k_case", SYNTHETIC_K)
@pytest.mark.parametrize("metric", METRICS)
def test_constant_real_column_is_skipped_with_a_note(metric, k_case):
    real, syn = _frames(SYNTHETIC_K[k_case])

    result = metric().compute(real, syn, _numeric_meta("x", "k"))
    without_k = metric().compute(real[["x"]], syn[["x"]], _numeric_meta("x"))

    assert result.per_column is not None and result.scalars is not None
    assert without_k.scalars is not None
    assert "k" not in result.per_column
    assert "column 'k' skipped (constant in the real data)" in result.notes
    assert result.scalars == pytest.approx(without_k.scalars)


@pytest.mark.parametrize("metric", METRICS)
def test_metric_is_skipped_when_every_real_column_is_constant(metric):
    df = pd.DataFrame({"k": np.full(100, 5.0)})

    result = metric().compute(df, df, _numeric_meta("k"))

    assert result.scalars is None
    assert result.skip_reason is not None
    assert "constant in the real data" in result.skip_reason
