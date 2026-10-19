import numpy as np
import pandas as pd

from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
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


def test_model_panel_infers_classification_for_string_target(real_df, syn_good_df, metadata):
    r = ModelPanelUtility(tasks=[UtilityTask(target="sex")]).compute(real_df, syn_good_df, metadata)
    assert "sex/random_forest" in r.per_column
    assert "task 'sex': task_type inferred as classification" in r.notes


def test_model_panel_unstratified_split_when_a_class_has_one_row(real_df, syn_good_df, metadata):
    real = real_df.copy()
    real.loc[real.index[0], "sex"] = "Other"
    r = ModelPanelUtility(tasks=[UtilityTask(target="sex")]).compute(real, syn_good_df, metadata)
    assert "sex/random_forest" in r.per_column
    assert any("split without stratification" in n for n in r.notes)


def test_model_panel_infers_classification_for_int_coded_target(real_df, syn_good_df, metadata):
    inferred = ModelPanelUtility(tasks=[UtilityTask(target="high_income")])
    explicit = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")]
    )
    assert (
        inferred.compute(real_df, syn_good_df, metadata).scalars
        == explicit.compute(real_df, syn_good_df, metadata).scalars
    )


def test_model_panel_non_positive_baseline_is_excluded(real_df, syn_good_df, metadata):
    from unittest import mock

    import syneva.utility.extended.model_panel as panel_mod

    # per model: (trtr, tstr). linear has no real-data skill; the others are fine.
    it = iter([-0.5, -2.0, 0.8, 0.4, 0.8, 0.4])
    with mock.patch.object(panel_mod, "_score", lambda *a, **k: next(it)):
        r = ModelPanelUtility(tasks=[UtilityTask(target="income", task_type="regression")]).compute(
            real_df, syn_good_df, metadata
        )
    assert "ratio" not in r.per_column["income/linear"]
    assert "ratio_linear" not in r.scalars
    assert r.scalars["score"] == 0.5
    assert any("income/linear" in n for n in r.notes)


def test_model_panel_single_class_synthetic_is_scored_in_auc(real_df, syn_good_df, metadata):
    syn = syn_good_df.copy()
    syn["high_income"] = 0
    r = ModelPanelUtility(
        tasks=[UtilityTask(target="high_income", task_type="classification")]
    ).compute(real_df, syn, metadata)
    assert r.per_column["high_income/random_forest"]["tstr"] == 0.5


def test_model_panel_skips_task_without_usable_features(real_df, syn_good_df, metadata):
    real = real_df.assign(signup=pd.Timestamp("2020-01-01"))
    syn = syn_good_df.assign(signup=pd.Timestamp("2020-01-01"))
    meta = Metadata(
        columns={**metadata.columns, "signup": ColumnMetadata("signup", ColumnType.DATETIME)}
    )
    tasks = [
        UtilityTask("high_income", "classification", features=["signup"]),
        UtilityTask("income", "regression"),
    ]
    r = ModelPanelUtility(tasks=tasks).compute(real, syn, meta)
    assert r.per_column is not None
    assert not any(k.startswith("high_income/") for k in r.per_column)
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


def test_model_panel_treats_none_and_pd_na_like_nan(real_df, syn_good_df, metadata):
    nan, none, nullable = _null_variants(syn_good_df)
    tasks = [UtilityTask("income", "regression")]
    results = [
        ModelPanelUtility(tasks=tasks).compute(real_df, s, metadata).per_column
        for s in (nan, none, nullable)
    ]
    assert results[0] is not None
    assert results[0] == results[1] == results[2]


def test_model_panel_dedupes_override_features(real_df, syn_good_df, metadata):
    def run(features):
        task = UtilityTask("income", "regression", features=features)
        return ModelPanelUtility(tasks=[task]).compute(real_df, syn_good_df, metadata).per_column

    assert run(["age", "age", "education"]) == run(["age", "education"])


def test_model_panel_ignores_real_values_of_feature_missing_in_synthetic(
    real_df, syn_good_df, metadata
):
    # An all-missing synthetic column is kept as a constant, so the synthetic-trained
    # model never uses it and the real values of that column cannot change TSTR
    syn = syn_good_df.assign(age=np.nan)
    shuffled = real_df.assign(age=np.random.default_rng(1).permutation(real_df["age"].to_numpy()))
    tasks = [UtilityTask("income", "regression")]

    def tstr(real):
        pc = ModelPanelUtility(tasks=tasks).compute(real, syn, metadata).per_column
        assert pc is not None
        return {k: v["tstr"] for k, v in pc.items()}

    assert tstr(real_df) == tstr(shuffled)


def test_model_panel_excludes_target_from_override(real_df, syn_good_df, metadata):
    task = UtilityTask("income", "regression", features=["age", "income"])
    r = ModelPanelUtility(tasks=[task]).compute(real_df, syn_good_df, metadata)
    assert any("'income'" in n and "ignored" in n for n in r.notes)
