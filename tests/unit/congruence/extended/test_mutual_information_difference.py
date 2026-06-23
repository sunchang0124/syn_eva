import numpy as np
import pandas as pd

from syneva.congruence.extended.mutual_information_difference import MutualInformationDifference
from syneva.core.metadata import Metadata


def test_identical_data_zero_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    df = pd.DataFrame(
        {"a": a, "b": a * 2 + rng.normal(scale=0.1, size=400), "c": rng.normal(size=400)}
    )
    r = MutualInformationDifference().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_mi_diff"] < 1e-9
    assert r.scalars["score"] > 0.99


def test_broken_dependency_raises_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    real = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.1, size=400)})
    syn = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    r = MutualInformationDifference().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_mi_diff"] > 0.1
    assert r.scalars["score"] < 0.9
