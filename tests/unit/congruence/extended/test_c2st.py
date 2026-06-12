import numpy as np
import pandas as pd

from syneva.congruence.extended.c2st import C2ST
from syneva.core.metadata import Metadata


def test_indistinguishable_data_auc_near_half():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert 0.45 <= r.scalars["c2st_auc"] <= 0.65


def test_separable_data_auc_near_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    syn = pd.DataFrame({"x": rng.normal(loc=5, size=500), "y": rng.normal(loc=5, size=500)})
    r = C2ST().compute(real, syn, Metadata.infer(real))
    assert r.scalars["c2st_auc"] > 0.95
