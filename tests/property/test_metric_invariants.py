import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import syneva  # registers metrics
from syneva.congruence.correlation_diff import CorrelationDifference
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata


@settings(max_examples=15, deadline=None)
@given(n=st.integers(min_value=50, max_value=200))
def test_identical_data_scores_high_on_congruence(n):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "sex": rng.choice(["M", "F"], n),
            "income": rng.normal(50_000, 8_000, n).round(2),
        }
    )
    rep = syneva.evaluate(df, df, tiers=("core",))
    assert rep.aggregated["congruence"] >= 0.95


@settings(max_examples=10, deadline=None)
@given(shift=st.floats(min_value=10.0, max_value=50.0))
def test_shifted_data_lowers_congruence(shift):
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)})
    syn = pd.DataFrame({"x": rng.normal(shift, 1, 200), "y": rng.normal(shift, 1, 200)})
    rep = syneva.evaluate(real, syn, tiers=("core",))
    assert rep.aggregated["congruence"] < 0.6


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
def test_correlation_difference_stays_in_unit_interval(seed, rho_real, rho_syn, shift):
    meta = Metadata(
        columns={
            "a": ColumnMetadata(name="a", dtype=ColumnType.NUMERIC),
            "b": ColumnMetadata(name="b", dtype=ColumnType.NUMERIC),
            "c": ColumnMetadata(name="c", dtype=ColumnType.CATEGORICAL),
            "d": ColumnMetadata(name="d", dtype=ColumnType.CATEGORICAL),
        }
    )
    result = CorrelationDifference().compute(
        _mixed_frame(seed, rho_real, shift), _mixed_frame(seed + 1, rho_syn, shift), meta
    )
    assert result.per_column is not None and result.scalars is not None
    for pair in result.per_column.values():
        assert 0.0 <= pair["abs_corr_diff"] <= 1.0
    mean = result.scalars["mean_abs_corr_diff"]
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= result.scalars["score"] <= 1.0
    assert result.scalars["score"] == pytest.approx(1.0 - mean)
