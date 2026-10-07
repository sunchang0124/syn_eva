import pandas as pd
import pytest

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


def test_dcr_holdout_baseline_flags_memorization():
    import pandas as pd

    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": [float(v) for v in range(100)]})
    syn = real.copy()  # verbatim copy => far closer to real than a disjoint holdout
    hold = pd.DataFrame({"x": [float(v) for v in range(1000, 1100)]})
    r = DCR(holdout=hold).compute(real, syn, Metadata.infer(real))
    assert "p05_dcr_holdout" in r.scalars
    assert r.scalars["score"] < 0.5  # synthetic far closer than holdout -> risk


def test_dcr_holdout_safe_when_synthetic_like_holdout():
    import pandas as pd

    from syneva.core.metadata import Metadata

    real = pd.DataFrame({"x": [float(v) for v in range(100)]})
    syn = pd.DataFrame({"x": [v + 500.0 for v in range(100)]})
    hold = pd.DataFrame({"x": [v + 500.0 for v in range(100)]})  # syn as far as holdout
    r = DCR(holdout=hold).compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] > 0.8


def _normal(n, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])


def _categoricals(n, seed, n_cols=15):
    import numpy as np

    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{i}": rng.choice(list("abcde"), size=n) for i in range(n_cols)})


@pytest.mark.parametrize("distance", ["euclidean", "gower"])
@pytest.mark.parametrize("make", [_normal, _categoricals])
def test_dcr_fresh_sample_from_same_distribution_scores_high(distance, make):
    # Regression for #18: without a holdout the score was min(1, p05 distance),
    # so this ideal case scored 0.015 (euclidean) / 0.0013 (gower) on 2-D data
    # and 1.0 / 0.33 on 15 categoricals.
    real, syn = make(500, 0), make(500, 1)
    r = DCR(distance=distance).compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] > 0.8


@pytest.mark.parametrize("distance", ["euclidean", "gower"])
def test_dcr_verbatim_copy_scores_zero(distance):
    real = _normal(500, 0)
    r = DCR(distance=distance).compute(real, real.copy(), Metadata.infer(real))
    assert r.scalars["score"] == 0.0


def test_dcr_reports_real_to_real_reference():
    real, syn = _normal(200, 0), _normal(200, 1)
    r = DCR().compute(real, syn, Metadata.infer(real))
    assert r.scalars["p05_dcr_real"] > 0.0
    assert r.scalars["score"] == pytest.approx(
        min(1.0, r.scalars["p05_dcr"] / r.scalars["p05_dcr_real"])
    )


def test_dcr_skipped_when_real_has_no_distance_baseline():
    # Every real row has an exact duplicate, so real->real p05 is 0; a copy
    # also sits at distance 0 and there is nothing to compare against.
    real = pd.DataFrame({"c": ["a", "b"] * 50})
    r = DCR().compute(real, real.copy(), Metadata.infer(real))
    assert r.scalars is None
    assert r.skip_reason


def test_dcr_far_synthetic_scores_one_when_real_baseline_is_zero():
    real = pd.DataFrame({"x": [0.0, 1.0] * 50})
    syn = pd.DataFrame({"x": [10.0, 11.0] * 50})
    r = DCR().compute(real, syn, Metadata.infer(real))
    assert r.scalars["score"] == 1.0


def test_dcr_skipped_with_single_real_row():
    real = pd.DataFrame({"x": [0.0]})
    syn = pd.DataFrame({"x": [1.0, 2.0]})
    r = DCR().compute(real, syn, Metadata.infer(real))
    assert r.scalars is None
    assert r.skip_reason
