import syneva
from syneva.utility.task import UtilityTask


def test_model_panel_runs_via_evaluate(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        tiers=("core", "extended"),
        run_utility=True,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    by = {r.spec.name: r for r in rep.results}
    assert "model_panel_utility" in by
    assert by["model_panel_utility"].error is None
    assert "ratio_spread" in by["model_panel_utility"].scalars


def test_model_panel_runs_via_full_preset(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        preset="full",
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    names = {r.spec.name for r in rep.results}
    assert "model_panel_utility" in names
