from unittest import mock

import numpy as np
import pandas as pd
import pytest

import syneva.utility.tstr as tstr_mod
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
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


def test_tstr_unstratified_split_when_a_class_has_one_row(real_df, syn_good_df, metadata):
    real = real_df.copy()
    real.loc[real.index[0], "sex"] = "Other"  # singleton class cannot be stratified
    r = TSTRSuite(tasks=[UtilityTask(target="sex")]).compute(real, syn_good_df, metadata)
    assert "utility_ratio_sex" in r.scalars
    assert any("split without stratification" in n for n in r.notes)


def test_tstr_unstratified_split_when_test_set_smaller_than_class_count(syn_good_df, metadata):
    # 30 rows, 15 classes of 2: a 30% test split holds 9 rows, too few for one per class
    real = syn_good_df.head(30).copy()
    real["sex"] = [f"c{i // 2}" for i in range(30)]
    # Make the class learnable from a feature, so the real-trained baseline has
    # positive skill and the task is not excluded as "TRTR <= 0" (see #64).
    real["age"] = [20 + i // 2 for i in range(30)]
    r = TSTRSuite(tasks=[UtilityTask(target="sex")]).compute(real, syn_good_df, metadata)
    assert "utility_ratio_sex" in r.scalars
    assert any("split without stratification" in n for n in r.notes)


def test_tstr_stratifies_when_classes_allow(real_df, syn_good_df, metadata):
    r = TSTRSuite(tasks=[UtilityTask(target="sex")]).compute(real_df, syn_good_df, metadata)
    assert not any("split without stratification" in n for n in r.notes)


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


def test_tstr_skips_task_without_usable_features(real_df, syn_good_df, metadata):
    # A datetime-only feature list used to reach the model as a 0-column matrix
    real = real_df.assign(signup=pd.Timestamp("2020-01-01"))
    syn = syn_good_df.assign(signup=pd.Timestamp("2020-01-01"))
    meta = Metadata(
        columns={**metadata.columns, "signup": ColumnMetadata("signup", ColumnType.DATETIME)}
    )
    tasks = [
        UtilityTask("high_income", "classification", features=["signup"]),
        UtilityTask("income", "regression"),
    ]
    r = TSTRSuite(tasks=tasks).compute(real, syn, meta)
    assert r.per_column is not None
    assert set(r.per_column) == {"income"}
    assert any("high_income" in n and "usable" in n for n in r.notes)


def _null_variants(syn):
    mask = syn.index % 3 != 0
    nan = syn.assign(education=syn["education"].mask(mask, np.nan))
    none = syn.copy()
    none["education"] = none["education"].astype(object).mask(mask, None)
    nullable = syn.assign(
        education=syn["education"].astype("string").mask(mask),
        age=syn["age"].astype("Int64"),
    )
    return nan, none, nullable


def test_tstr_treats_none_and_pd_na_like_nan(real_df, syn_good_df, metadata):
    nan, none, nullable = _null_variants(syn_good_df)
    tasks = [UtilityTask("income", "regression")]
    results = [
        TSTRSuite(tasks=tasks).compute(real_df, s, metadata).per_column
        for s in (nan, none, nullable)
    ]
    assert results[0] is not None
    assert results[0] == results[1] == results[2]


def test_tstr_dedupes_override_features(real_df, syn_good_df, metadata):
    def run(features):
        task = UtilityTask("income", "regression", features=features)
        return TSTRSuite(tasks=[task]).compute(real_df, syn_good_df, metadata).per_column

    assert run(["age", "age", "education"]) == run(["age", "education"])


def test_tstr_ignores_real_values_of_feature_missing_in_synthetic(real_df, syn_good_df, metadata):
    # An all-missing synthetic column is kept as a constant, so the synthetic-trained
    # model never uses it and the real values of that column cannot change TSTR
    syn = syn_good_df.assign(age=np.nan)
    shuffled = real_df.assign(age=np.random.default_rng(1).permutation(real_df["age"].to_numpy()))
    tasks = [UtilityTask("income", "regression")]

    def tstr(real):
        pc = TSTRSuite(tasks=tasks).compute(real, syn, metadata).per_column
        assert pc is not None
        return {k: v["tstr"] for k, v in pc.items()}

    assert tstr(real_df) == tstr(shuffled)


def test_tstr_excludes_target_from_override(real_df, syn_good_df, metadata):
    task = UtilityTask("income", "regression", features=["age", "income"])
    r = TSTRSuite(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert any("'income'" in n and "ignored" in n for n in r.notes)
