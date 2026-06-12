import syneva
import syneva.utility.tstr
from syneva.utility.task import UtilityTask


def test_utility_off_by_default(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata)
    by_name = {r.spec.name: r for r in rep.results}
    # TSTRSuite is registered but not run when no utility_tasks + run_utility=False
    assert "tstr_suite" not in by_name


def test_run_utility_with_explicit_task(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
        run_utility=True,
    )
    by_name = {r.spec.name: r for r in rep.results}
    assert "tstr_suite" in by_name
    assert by_name["tstr_suite"].scalars["score"] > 0.0


def test_auto_suggest_runs_when_flag_set(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, run_utility=True)
    by_name = {r.spec.name: r for r in rep.results}
    assert "tstr_suite" in by_name
    # run_info should record what was auto-suggested
    assert any("auto-suggested" in w for w in rep.run_info.warnings)
