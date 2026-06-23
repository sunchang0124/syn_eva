from syneva.compliance.extended.mia import MembershipInferenceAttack


def test_mia_returns_auc_in_range(real_df, syn_good_df, metadata):
    r = MembershipInferenceAttack().compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["mia_auc"] <= 1.0
    assert 0.0 <= r.scalars["score"] <= 1.0


def test_mia_with_holdout_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=3)
    r = MembershipInferenceAttack(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["mia_auc"] <= 1.0
    assert 0.0 <= r.scalars["score"] <= 1.0
