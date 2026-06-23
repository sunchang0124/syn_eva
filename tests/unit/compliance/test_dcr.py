import pandas as pd

from syneva.compliance.dcr import DCR
from syneva.core.metadata import Metadata


def test_far_synthetic_high_dcr():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(1000, 1100))})
    r = DCR().compute(real, syn, Metadata.infer(real))
    assert r.scalars["median_dcr"] > 1.0
    assert r.scalars["score"] > 0.5


def test_near_synthetic_low_dcr(real_df, syn_leaky_df, metadata):
    r = DCR().compute(real_df, syn_leaky_df, metadata)
    assert r.scalars["median_dcr"] < 0.5
    assert r.scalars["score"] < 0.5


def test_dcr_gower_backend_runs():
    import pandas as pd

    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": list(range(100)), "c": ["a", "b"] * 50})
    syn = pd.DataFrame({"x": list(range(1000, 1100)), "c": ["a", "b"] * 50})
    r = DCR(distance="gower").compute(real, syn, Metadata.infer(real))
    assert 0.0 <= r.scalars["score"] <= 1.0
    assert r.scalars["median_dcr"] >= 0.0
