import math

import pytest

from syneva.benchmark.engine import BenchmarkResult, benchmark
from syneva.core.errors import SynevaError


def test_runs_each_candidate(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    assert isinstance(res, BenchmarkResult)
    assert set(res.reports) == {"good", "shifted"}
    assert "good" in res.score_matrix and "shifted" in res.score_matrix


def test_overall_is_equal_weight_c_mean(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    cs = res.c_scores["good"]
    expected = sum(cs.values()) / len(cs)
    assert math.isclose(res.overall["good"], expected, rel_tol=1e-9)


def test_metrics_and_cdims_are_sorted_unions(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    assert res.metrics() == sorted(res.metrics())
    assert res.c_dims() == sorted(res.c_dims())


def test_empty_candidates_raises(real_df, metadata):
    with pytest.raises(SynevaError, match="at least one candidate"):
        benchmark(real_df, {}, metadata)


def test_schema_mismatch_candidate_raises(real_df, syn_good_df, metadata):
    bad = syn_good_df.drop(columns=[syn_good_df.columns[0]])
    with pytest.raises(SynevaError, match="bad_cand"):
        benchmark(real_df, {"ok": syn_good_df, "bad_cand": bad}, metadata)


def test_preset_forwarded_to_all(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(
        real_df,
        {"good": syn_good_df, "shifted": syn_shifted_df},
        metadata,
        preset="privacy",
    )
    for name in ("good", "shifted"):
        assert set(res.c_scores[name]) == {"compliance"}


def test_ranking_better_candidate_wins_all_modes(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    for mode in ("absolute", "linear", "normal", "quantile"):
        ranked = res.ranking(normalization=mode)
        assert ranked[0][1] == "good", f"mode={mode}"
        assert [r[0] for r in ranked] == [1, 2]


def test_ranking_by_single_c(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    ranked = res.ranking(by="compliance")
    assert {r[1] for r in ranked} == {"good", "shifted"}


def test_ranking_unknown_by_raises(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df}, metadata)
    with pytest.raises(SynevaError, match="by"):
        res.ranking(by="not_a_thing")


def test_single_candidate_ranks_first_with_note(real_df, syn_good_df, metadata):
    res = benchmark(real_df, {"only": syn_good_df}, metadata)
    ranked = res.ranking(normalization="linear")
    assert ranked == [(1, "only", ranked[0][2])]


def test_roundtrip_dict_preserves_ranking(real_df, syn_good_df, syn_shifted_df, metadata):
    res = benchmark(real_df, {"good": syn_good_df, "shifted": syn_shifted_df}, metadata)
    res2 = BenchmarkResult.from_dict(res.to_dict())
    assert res2.ranking() == res.ranking()
    assert res2.overall == res.overall
