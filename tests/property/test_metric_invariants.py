import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

import syneva  # registers metrics


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
