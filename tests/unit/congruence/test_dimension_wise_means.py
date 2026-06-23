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
