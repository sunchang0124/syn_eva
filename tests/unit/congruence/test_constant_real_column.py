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


def _real_and_synthetic(const_syn: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real: a normal column ``x`` and a column ``const`` that is always 5.0.

    Synthetic: ``x`` with a little noise, and ``const_syn`` as ``const``.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    real = pd.DataFrame({"x": x, "const": np.full(N, 5.0)})
    syn = pd.DataFrame({"x": x + rng.normal(scale=0.1, size=N), "const": const_syn})
    return real, syn


SYNTHETIC_CONSTANT_COLUMN = {
    "noise": 5.0 + np.random.default_rng(1).normal(scale=0.01, size=N),
    "exact match": np.full(N, 5.0),
    "other constant": np.full(N, 6.0),
}


@pytest.mark.parametrize("synthetic_case", SYNTHETIC_CONSTANT_COLUMN)
@pytest.mark.parametrize("metric", METRICS)
def test_constant_real_column_is_skipped_with_a_note(metric, synthetic_case):
    real, syn = _real_and_synthetic(SYNTHETIC_CONSTANT_COLUMN[synthetic_case])

    result = metric().compute(real, syn, _numeric_meta("x", "const"))
    without_const = metric().compute(real[["x"]], syn[["x"]], _numeric_meta("x"))

    assert result.per_column is not None and result.scalars is not None
    assert without_const.scalars is not None
    assert "const" not in result.per_column
    assert "column 'const' skipped (constant in the real data)" in result.notes
    assert result.scalars == pytest.approx(without_const.scalars)


@pytest.mark.parametrize("metric", METRICS)
def test_metric_is_skipped_when_every_real_column_is_constant(metric):
    df = pd.DataFrame({"const": np.full(100, 5.0)})

    result = metric().compute(df, df, _numeric_meta("const"))

    assert result.scalars is None
    assert result.skip_reason is not None
    assert "constant in the real data" in result.skip_reason
