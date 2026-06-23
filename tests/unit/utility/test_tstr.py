from syneva.utility.task import UtilityTask
from syneva.utility.tstr import TSTRSuite


def test_tstr_scores_present(real_df, syn_good_df, metadata):
    suite = TSTRSuite(tasks=[UtilityTask(target="high_income", task_type="classification")])
    r = suite.compute(real_df, syn_good_df, metadata)
    keys = set(r.scalars.keys())
    assert {"trtr_high_income", "tstr_high_income", "utility_ratio_high_income"} <= keys
    assert r.scalars["utility_ratio_high_income"] > 0.5


def test_tstr_regression_task(real_df, syn_good_df, metadata):
    suite = TSTRSuite(tasks=[UtilityTask(target="income", task_type="regression")])
    r = suite.compute(real_df, syn_good_df, metadata)
    keys = set(r.scalars.keys())
    assert "tstr_income" in keys


def test_tstr_uses_holdout_as_test_set(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=7)
    suite = TSTRSuite(
        tasks=[UtilityTask(target="high_income", task_type="classification")],
        holdout=holdout,
    )
    r = suite.compute(real_df, syn_good_df, metadata)
    assert "utility_ratio_high_income" in r.scalars
    assert r.scalars["utility_ratio_high_income"] >= 0.0
