import pytest

from syneva.utility.extended.discriminative import DiscriminativeScore
from syneva.utility.extended.feature_importance import FeatureImportanceCorrelation
from syneva.utility.extended.multi_target import MultiTargetUtility
from syneva.utility.task import UtilityTask


def test_multi_target_runs_all_columns(real_df, syn_good_df, metadata):
    r = MultiTargetUtility().compute(real_df, syn_good_df, metadata)
    assert r.scalars["score"] >= 0.0
    assert r.per_column  # at least one auto-suggested task


def test_feature_importance_correlation_returns_value(real_df, syn_good_df, metadata):
    r = FeatureImportanceCorrelation(tasks=[UtilityTask("high_income")]).compute(
        real_df, syn_good_df, metadata
    )
    assert -1.0 <= r.scalars["spearman_rho"] <= 1.0


def test_feature_importance_averages_over_tasks(real_df, syn_good_df, metadata):
    tasks = [UtilityTask("high_income", "classification"), UtilityTask("age", "regression")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert set(r.per_column) == {"high_income", "age"}
    rhos = [r.per_column[t]["spearman_rho"] for t in ("high_income", "age")]
    assert r.scalars["spearman_rho"] == pytest.approx(sum(rhos) / 2)
    assert r.scalars["score"] == pytest.approx((r.scalars["spearman_rho"] + 1) / 2)


def test_feature_importance_honours_task_features(real_df, syn_good_df, metadata):
    task = UtilityTask("high_income", "classification", features=["age", "hours_per_week"])
    r = FeatureImportanceCorrelation(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert r.per_column["high_income"]["n_features"] == 2


def test_feature_importance_notes_missing_target(real_df, syn_good_df, metadata):
    tasks = [UtilityTask("nope", "classification"), UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert set(r.per_column) == {"high_income"}
    assert any("nope" in n for n in r.notes)


def test_feature_importance_without_tasks_is_skipped(real_df, syn_good_df, metadata):
    r = FeatureImportanceCorrelation().compute(real_df, syn_good_df, metadata)
    assert r.scalars is None
    assert r.skip_reason


def test_discriminative_score_returns_auc(real_df, syn_good_df, metadata):
    r = DiscriminativeScore().compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["auc"] <= 1.0


def test_feature_importance_undefined_rho_falls_back_to_zero(real_df, syn_shifted_df, metadata):
    # The shifted fixture's synthetic high_income is constant, so its model has
    # all-zero importances and Spearman rho is undefined (NaN).
    tasks = [UtilityTask("high_income", "classification")]
    r = FeatureImportanceCorrelation(tasks=tasks).compute(real_df, syn_shifted_df, metadata)
    assert r.scalars["spearman_rho"] == 0.0
    assert r.scalars["score"] == 0.5
    assert any("undefined" in n for n in r.notes)
