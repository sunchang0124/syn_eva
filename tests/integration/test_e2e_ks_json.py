# tests/integration/test_e2e_ks_json.py
import json

import syneva.congruence.ks  # noqa: F401  ensure registration
from syneva import evaluate


def test_evaluate_with_ks_and_round_trip_json(real_df, syn_good_df, metadata, tmp_path):
    rep = evaluate(real_df, syn_good_df, metadata, tiers=("core",), data_type="static")
    by_name = {r.spec.name: r for r in rep.results}
    assert "ks_statistic" in by_name
    assert by_name["ks_statistic"].scalars["score"] > 0.8

    out = tmp_path / "scorecard.json"
    rep.to_json(out)
    loaded = json.loads(out.read_text())
    assert any(r["spec"]["name"] == "ks_statistic" for r in loaded["results"])
    assert "library_versions" in loaded["run_info"]


def test_aggregated_returns_congruence(real_df, syn_good_df, metadata):
    rep = evaluate(real_df, syn_good_df, metadata)
    assert "congruence" in rep.aggregated
    assert 0.0 <= rep.aggregated["congruence"] <= 1.0
