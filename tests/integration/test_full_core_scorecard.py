import json

import syneva


def test_core_scorecard_on_good_synth(real_df, syn_good_df, metadata, tmp_path):
    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        tiers=("core",),
        run_utility=True,
        utility_tasks=[syneva.UtilityTask(target="high_income", task_type="classification")],
    )
    cs = set(rep.by_c.keys())
    assert {"congruence", "coverage", "compliance", "utility"} <= cs

    # Sanity: good synth should average well on Congruence
    assert rep.aggregated["congruence"] > 0.6

    rep.to_html(tmp_path / "scorecard.html")
    rep.to_json(tmp_path / "scorecard.json")
    assert (tmp_path / "scorecard.html").stat().st_size > 1000
    json.loads((tmp_path / "scorecard.json").read_text())


def test_core_scorecard_on_leaky_synth_fires_privacy(real_df, syn_leaky_df, metadata):
    rep = syneva.evaluate(real_df, syn_leaky_df, metadata, tiers=("core",))
    assert rep.aggregated["compliance"] < 0.6


def test_new_fidelity_metrics_present(real_df, syn_good_df, metadata):
    import syneva

    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core", "extended"))
    names = {r.spec.name for r in rep.results}
    for m in [
        "dimension_wise_means",
        "ci_overlap",
        "hellinger",
        "quantile_mse",
        "mutual_information_difference",
        "mmd",
        "nn_adversarial_accuracy",
    ]:
        assert m in names, f"{m} not in evaluate() results"
        result = next(r for r in rep.results if r.spec.name == m)
        assert result.error is None, f"{m} errored: {result.error}"
