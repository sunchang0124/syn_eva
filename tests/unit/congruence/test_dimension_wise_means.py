import pandas as pd

from syneva.congruence.dimension_wise_means import DimensionWiseMeans
from syneva.core.metadata import Metadata


def test_identical_data_scores_high():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    r = DimensionWiseMeans().compute(df, df, Metadata.infer(df))
    assert r.scalars["score"] > 0.95
    assert r.scalars["mean_abs_std_diff"] < 0.05


def test_mean_shift_lowers_score():
    real = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    syn = pd.DataFrame({"x": [11.0, 12.0, 13.0, 14.0, 15.0]})
    r = DimensionWiseMeans().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_abs_std_diff"] > 1.0
    assert r.scalars["score"] < 0.5


def test_identical_constant_column_no_nan():
    import math

    df = pd.DataFrame({"x": [5.0] * 100})
    r = DimensionWiseMeans().compute(df, df, Metadata.infer(df))
    # constant column: std=0 over 100 rows, |mean diff|=0 -> std_mean_diff 0, score 1
    assert r.per_column["x"]["std_mean_diff"] == 0.0
    assert not math.isnan(r.scalars["score"])
    assert r.scalars["score"] == 1.0
