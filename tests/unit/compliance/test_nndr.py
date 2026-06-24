from syneva.compliance.nndr import NNDR


def test_far_synthetic_balanced_ratio(real_df, syn_good_df, metadata):
    r = NNDR().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] > 0.3


def test_leaky_synthetic_low_ratio(real_df, syn_leaky_df, metadata):
    r = NNDR().compute(real_df, syn_leaky_df, metadata)
    assert r.scalars["nndr_median"] < 0.3
    assert r.scalars["score"] < 0.5


def test_nndr_gower_backend_runs(real_df, syn_good_df, metadata):
    r = NNDR(distance="gower").compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["score"] <= 1.0


def test_nndr_holdout_baseline_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=5)
    r = NNDR(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert "nndr_median_holdout" in r.scalars
    assert 0.0 <= r.scalars["score"] <= 1.0
