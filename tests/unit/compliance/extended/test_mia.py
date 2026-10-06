from syneva.compliance.extended.mia import MembershipInferenceAttack


def test_mia_without_holdout_is_skipped(real_df, syn_good_df, metadata):
    # Without a holdout there are no true non-members, so the attack has no signal.
    # It must be skipped (no scalars), not reported as a perfect privacy score.
    r = MembershipInferenceAttack().compute(real_df, real_df.copy(), metadata)
    assert r.scalars is None
    assert r.error is None
    assert r.skip_reason is not None
    assert "holdout" in r.skip_reason


def test_mia_with_holdout_runs(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.4, random_state=3)
    r = MembershipInferenceAttack(holdout=holdout).compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["mia_auc"] <= 1.0
    assert 0.0 <= r.scalars["score"] <= 1.0


def test_mia_with_holdout_detects_verbatim_copy(real_df, metadata):
    # Disjoint split: members are copied verbatim, non-members were never seen.
    holdout = real_df.sample(frac=0.3, random_state=3)
    members = real_df.drop(index=holdout.index)
    r = MembershipInferenceAttack(holdout=holdout).compute(members, members.copy(), metadata)
    assert r.scalars["mia_auc"] > 0.9
    assert r.scalars["score"] < 0.2
