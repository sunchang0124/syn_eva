from syneva.core.metadata import Metadata
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


def test_tstr_infers_classification_for_string_target(real_df, syn_good_df, metadata):
    r = TSTRSuite(tasks=[UtilityTask(target="sex")]).compute(real_df, syn_good_df, metadata)
    assert "utility_ratio_sex" in r.scalars
    assert "task 'sex': task_type inferred as classification" in r.notes


def test_tstr_handles_target_missing_from_metadata(real_df, syn_good_df, metadata):
    meta = Metadata(columns={n: c for n, c in metadata.columns.items() if n != "sex"})
    r = TSTRSuite(tasks=[UtilityTask(target="sex")]).compute(real_df, syn_good_df, meta)
    assert "utility_ratio_sex" in r.scalars


def test_tstr_infers_classification_for_int_coded_target(real_df, syn_good_df, metadata):
    # high_income is 0/1 int64 but declared categorical in the metadata
    inferred = TSTRSuite(tasks=[UtilityTask(target="high_income")])
    explicit = TSTRSuite(tasks=[UtilityTask(target="high_income", task_type="classification")])
    assert (
        inferred.compute(real_df, syn_good_df, metadata).scalars
        == explicit.compute(real_df, syn_good_df, metadata).scalars
    )
