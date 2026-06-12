import numpy as np
import pandas as pd

from syneva.core.metadata import Metadata
from syneva.coverage.extended.alpha_precision import AlphaPrecisionBetaRecall


def test_identical_distribution_close_to_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    r = AlphaPrecisionBetaRecall().compute(df, df, Metadata.infer(df))
    assert r.scalars["alpha_precision"] > 0.7
    assert r.scalars["beta_recall"] > 0.7


def test_far_distribution_low_values():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    syn = pd.DataFrame({"x": rng.normal(loc=20, size=300), "y": rng.normal(loc=20, size=300)})
    r = AlphaPrecisionBetaRecall().compute(real, syn, Metadata.infer(real))
    assert r.scalars["alpha_precision"] < 0.2
    assert r.scalars["beta_recall"] < 0.2
