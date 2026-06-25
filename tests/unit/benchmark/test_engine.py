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
