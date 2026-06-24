import pytest

import syneva
from syneva.core.errors import SynevaError
from syneva.utility.task import UtilityTask


def test_holdout_runs_end_to_end(real_df, syn_good_df, metadata):
    holdout = real_df.sample(frac=0.3, random_state=2)
    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        tiers=("core", "extended"),
        holdout=holdout,
        run_utility=True,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert by["dcr"].error is None and "p05_dcr_holdout" in by["dcr"].scalars
    assert by["mia_auc"].error is None
    assert by["tstr_suite"].error is None


def test_holdout_schema_mismatch_raises(real_df, syn_good_df, metadata):
    bad = real_df.drop(columns=[real_df.columns[0]])
    with pytest.raises(SynevaError, match="holdout"):
        syneva.evaluate(real_df, syn_good_df, metadata, holdout=bad)
