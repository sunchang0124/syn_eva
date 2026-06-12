from syneva.coverage.extended.authenticity import Authenticity


def test_leaky_synthetic_low_authenticity(real_df, syn_leaky_df, metadata):
    r = Authenticity().compute(real_df, syn_leaky_df, metadata)
    assert r.scalars["authenticity"] < 0.5


def test_diverse_synthetic_high_authenticity(real_df, syn_good_df, metadata):
    r = Authenticity().compute(real_df, syn_good_df, metadata)
    assert r.scalars["authenticity"] > 0.5
