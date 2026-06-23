import numpy as np
import pandas as pd

from syneva.core.metadata import Metadata
from syneva.coverage.extended.nn_adversarial_accuracy import NNAdversarialAccuracy


def test_indistinguishable_data_aa_near_half():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    syn = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    r = NNAdversarialAccuracy().compute(real, syn, Metadata.infer(real))
    assert 0.4 <= r.scalars["nn_adversarial_accuracy"] <= 0.6
    assert r.scalars["score"] > 0.7


def test_separated_data_aa_near_one():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=400), "y": rng.normal(size=400)})
    syn = pd.DataFrame({"x": rng.normal(loc=20, size=400), "y": rng.normal(loc=20, size=400)})
    r = NNAdversarialAccuracy().compute(real, syn, Metadata.infer(real))
    assert r.scalars["nn_adversarial_accuracy"] > 0.9
    assert r.scalars["score"] < 0.3


def test_nnaa_gower_backend_runs():
    import numpy as np
    import pandas as pd

    from syneva.core.metadata import Metadata

    rng = np.random.default_rng(0)
    real = pd.DataFrame({"x": rng.normal(size=80), "c": ["a", "b"] * 40})
    syn = pd.DataFrame({"x": rng.normal(size=80), "c": ["a", "b"] * 40})
    r = NNAdversarialAccuracy(distance="gower").compute(real, syn, Metadata.infer(real))
    assert 0.0 <= r.scalars["score"] <= 1.0
