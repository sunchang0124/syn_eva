import numpy as np
import pandas as pd

from syneva.congruence.quantile_mse import QuantileMSE
from syneva.core.metadata import Metadata


def test_identical_data_low_qmse():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    r = QuantileMSE().compute(df, df, Metadata.infer(df))
    assert r.scalars["mean_quantile_mse"] < 1e-6
    assert r.scalars["score"] > 0.99


def test_tail_shift_raises_qmse():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(loc=0, scale=1, size=500)})
    syn = pd.DataFrame({"x": rng.normal(loc=0, scale=4, size=500)})
    r = QuantileMSE().compute(real, syn, Metadata.infer(real))
    assert r.scalars["mean_quantile_mse"] > 0.1
    assert r.scalars["score"] < 0.9
