from unittest import mock

import pytest

import syneva.utility.tstr as tstr_mod
from syneva.core.metadata import Metadata
from syneva.utility.task import UtilityTask
from syneva.utility.tstr import TSTRSuite, utility_ratio


def _fake_scores(*scores):
    """Patch _score to return trtr, tstr, trtr, tstr, ... in task order."""
    it = iter(scores)
    return mock.patch.object(tstr_mod, "_score", lambda *a, **k: next(it))


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


@pytest.mark.parametrize(
    ("trtr", "tstr", "expected"),
    [
        (0.8, 0.4, 0.5),
        (0.5, 0.9, 1.0),  # synthetic beats real: clipped
        (0.8, -0.3, 0.0),  # negative R^2 on the synthetic side: clipped
        (0.0, 0.5, None),  # no real-data skill: ratio undefined
        (-0.5, -2.0, None),  # previously 4.0 -> clipped to a perfect 1.0
    ],
)
def test_utility_ratio_is_clipped_and_sign_safe(trtr, tstr, expected):
    assert utility_ratio(trtr, tstr) == expected


def test_tstr_non_positive_baseline_is_excluded_not_perfect(real_df, syn_good_df, metadata):
    with _fake_scores(-0.5, -2.0):
        r = TSTRSuite(tasks=[UtilityTask(target="income", task_type="regression")]).compute(
            real_df, syn_good_df, metadata
        )
    assert r.scalars is None
    assert r.skip_reason
    assert any("income" in n and "TRTR" in n for n in r.notes)


def test_tstr_excluded_task_keeps_raw_scores(real_df, syn_good_df, metadata):
    tasks = [
        UtilityTask(target="income", task_type="regression"),
        UtilityTask(target="age", task_type="regression"),
    ]
    with _fake_scores(-0.5, -2.0, 0.8, 0.4):
        r = TSTRSuite(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert r.scalars["trtr_income"] == -0.5
    assert "utility_ratio_income" not in r.scalars
    assert "ratio" not in r.per_column["income"]
    assert r.scalars["score"] == pytest.approx(0.5)


def test_tstr_clips_each_ratio_before_averaging(real_df, syn_good_df, metadata):
    tasks = [
        UtilityTask(target="income", task_type="regression"),
        UtilityTask(target="age", task_type="regression"),
    ]
    # per-task ratios 0.2 and 1.8: the old unclipped mean hid the broken task as 1.0
    with _fake_scores(1.0, 0.2, 0.5, 0.9):
        r = TSTRSuite(tasks=tasks).compute(real_df, syn_good_df, metadata)
    assert r.per_column["age"]["ratio"] == 1.0
    assert r.scalars["score"] == pytest.approx(0.6)


def test_tstr_single_class_synthetic_is_scored_in_auc(real_df, syn_good_df, metadata):
    syn = syn_good_df.copy()
    syn["high_income"] = 0  # synthetic lost the positive class
    r = TSTRSuite(tasks=[UtilityTask(target="high_income", task_type="classification")]).compute(
        real_df, syn, metadata
    )
    # a constant predictor has AUC 0.5, not the majority-class accuracy
    assert r.scalars["tstr_high_income"] == pytest.approx(0.5)
    assert r.scalars["utility_ratio_high_income"] < 0.6
