import numpy as np

from syneva.utility.extended.model_panel import ModelPanelUtility
from syneva.utility.task import UtilityTask


def test_panel_reports_per_model_and_spread(real_df, syn_good_df, metadata):
    m = ModelPanelUtility(tasks=[UtilityTask(target="high_income", task_type="classification")])
    r = m.compute(real_df, syn_good_df, metadata)
    for key in (
        "score",
        "utility_ratio_mean",
        "ratio_linear",
        "ratio_random_forest",
        "ratio_hist_gbdt",
        "ratio_spread",
    ):
        assert key in r.scalars
        assert np.isfinite(r.scalars[key])
    assert 0.0 <= r.scalars["score"] <= 1.0
    assert "high_income/linear" in r.per_column
    assert "high_income/hist_gbdt" in r.per_column


def test_panel_identical_data_ratios_near_one(real_df, metadata):
    m = ModelPanelUtility(tasks=[UtilityTask(target="high_income", task_type="classification")])
    r = m.compute(real_df, real_df, metadata)  # synthetic == real
    assert r.scalars["score"] > 0.8
    assert r.scalars["ratio_spread"] < 0.3


def test_panel_no_tasks_is_skipped(real_df, syn_good_df, metadata):
    m = ModelPanelUtility(tasks=[])
    r = m.compute(real_df, syn_good_df, metadata)
    assert r.scalars is None
    assert r.skip_reason


def test_panel_runs_with_holdout(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=4)
    m = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")],
        holdout=holdout,
    )
    r = m.compute(real_df, syn_good_df, metadata)
    assert 0.0 <= r.scalars["score"] <= 1.0
