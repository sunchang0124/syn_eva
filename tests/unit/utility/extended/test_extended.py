from syneva.utility.extended.discriminative import DiscriminativeScore
from syneva.utility.extended.feature_importance import FeatureImportanceCorrelation
from syneva.utility.extended.multi_target import MultiTargetUtility


def test_multi_target_runs_all_columns(real_df, syn_good_df, metadata):
    r = MultiTargetUtility().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] >= 0.0
    assert r.per_column  # at least one auto-suggested task


def test_feature_importance_correlation_returns_value(real_df, syn_good_df, metadata):
    r = FeatureImportanceCorrelation(target="high_income").compute(real_df, syn_good_df, metadata)
    assert -1.0 <= r.scalars["spearman_rho"] <= 1.0


def test_discriminative_score_returns_auc(real_df, syn_good_df, metadata):
    r = DiscriminativeScore().compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["auc"] <= 1.0
