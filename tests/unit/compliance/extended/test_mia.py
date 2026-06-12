from syneva.compliance.extended.mia import MembershipInferenceAttack


def test_mia_returns_auc_in_range(real_df, syn_good_df, metadata):
    r = MembershipInferenceAttack().compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["mia_auc"] <= 1.0
    assert 0.0 <= r.scalars["score"] <= 1.0
