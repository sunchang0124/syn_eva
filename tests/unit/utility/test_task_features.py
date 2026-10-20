"""Feature selection and null handling shared by TSTR and the model panel."""

import numpy as np
import pytest

from syneva.utility.extended.model_panel import ModelPanelUtility
from syneva.utility.task import UtilityTask
from syneva.utility.tstr import TSTRSuite

# TSTR keys per_column by target; the panel by "target/model"
_METRICS = pytest.mark.parametrize("metric", [TSTRSuite, ModelPanelUtility])


def _run(metric, real, syn, meta, task):
    return metric(tasks=[task]).compute(real, syn, meta)


@_METRICS
def test_skips_task_without_usable_features(metric, with_datetime):
    # A datetime-only feature list used to reach the model as a 0-column matrix
    real, syn, meta = with_datetime
    tasks = [
        UtilityTask("high_income", "classification", features=["signup"]),
        UtilityTask("income", "regression"),
    ]
    r = metric(tasks=tasks).compute(real, syn, meta)
    assert r.per_column is not None
    assert {k.split("/")[0] for k in r.per_column} == {"income"}
    assert any("high_income" in n and "usable" in n for n in r.notes)


@_METRICS
def test_treats_none_and_pd_na_like_nan(metric, real_df, metadata, null_variants):
    task = UtilityTask("income", "regression")
    results = [_run(metric, real_df, s, metadata, task).per_column for s in null_variants]
    assert results[0] is not None
    assert results[0] == results[1] == results[2]


@_METRICS
def test_dedupes_override_features(metric, real_df, syn_good_df, metadata):
    def run(features):
        task = UtilityTask("income", "regression", features=features)
        return _run(metric, real_df, syn_good_df, metadata, task).per_column

    assert run(["age", "age", "education"]) == run(["age", "education"])


@_METRICS
def test_excludes_target_from_override(metric, real_df, syn_good_df, metadata):
    def run(features):
        task = UtilityTask("income", "regression", features=features)
        return _run(metric, real_df, syn_good_df, metadata, task)

    with_target = run(["age", "income"])
    assert with_target.per_column is not None
    assert with_target.per_column == run(["age"]).per_column
    assert any("target" in n and "ignored" in n for n in with_target.notes)


@_METRICS
def test_ignores_real_values_of_feature_missing_in_synthetic(
    metric, real_df, syn_good_df, metadata
):
    # An all-missing synthetic column is kept as a constant, so the synthetic-trained
    # model never uses it and the real values of that column cannot change TSTR
    syn = syn_good_df.assign(age=np.nan)
    shuffled = real_df.assign(age=np.random.default_rng(1).permutation(real_df["age"].to_numpy()))
    task = UtilityTask("income", "regression")

    def tstr(real):
        pc = _run(metric, real, syn, metadata, task).per_column
        assert pc is not None
        return {k: v["tstr"] for k, v in pc.items()}

    assert tstr(real_df) == tstr(shuffled)
